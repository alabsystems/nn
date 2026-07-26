// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Spectral (frequency-domain) comparison for audio tensors.
//!
//! Provides STFT-based metrics — log-spectral distance, spectral convergence,
//! per-bin magnitude difference, and phase coherence — for parity testing of
//! audio models where time-domain element-wise comparison is insufficient.
//!
//! Gated behind the `spectral` feature flag. Depends on `rustfft` for FFT.

use crate::error::ReftestError;

// STFT computation (WindowFn, StftConfig, StftResult, stft_magnitude, stft_full)
// extracted to stay under the 500-line limit (Part of #1575).
#[path = "spectral_stft.rs"]
mod stft;
pub use stft::{stft_full, stft_magnitude, StftConfig, StftResult, WindowFn};

// ---------------------------------------------------------------------------
// Spectral comparison configuration and results
// ---------------------------------------------------------------------------

/// Thresholds for spectral comparison metrics.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SpectralConfig {
    /// STFT parameters.
    pub stft: StftConfig,
    /// Maximum log-spectral distance (dB). Default: 1.0.
    pub max_lsd_db: f32,
    /// Maximum spectral convergence. Default: 0.01.
    pub max_spectral_convergence: f32,
    /// Minimum phase coherence. Default: 0.95.
    pub min_phase_coherence: f32,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            stft: StftConfig::default(),
            max_lsd_db: 1.0,
            max_spectral_convergence: 0.01,
            min_phase_coherence: 0.95,
        }
    }
}

impl SpectralConfig {
    /// Create a `SpectralConfig` with custom thresholds.
    #[must_use]
    pub fn new(max_lsd_db: f32, max_spectral_convergence: f32, min_phase_coherence: f32) -> Self {
        Self {
            stft: StftConfig::default(),
            max_lsd_db,
            max_spectral_convergence,
            min_phase_coherence,
        }
    }

    /// Override STFT parameters.
    #[must_use]
    pub fn with_stft(mut self, stft: StftConfig) -> Self {
        self.stft = stft;
        self
    }
}

/// Result of a spectral comparison between two audio signals.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpectralComparison {
    /// Log-spectral distance (dB). Lower = better. < 1.0 dB = excellent.
    pub log_spectral_distance_db: f32,
    /// Spectral convergence: `||S_ref - S_cand||_F / ||S_ref||_F`. Lower = better.
    pub spectral_convergence: f32,
    /// Per-frequency-bin maximum absolute magnitude difference (dB).
    pub max_magnitude_diff_db: f32,
    /// Mean magnitude difference across all bins (dB).
    pub mean_magnitude_diff_db: f32,
    /// Phase coherence: `mean(cos(phase_ref - phase_cand))`. 1.0 = identical phase.
    pub phase_coherence: f32,
    /// Whether comparison passed all spectral gates.
    pub passed: bool,
}

impl std::fmt::Display for SpectralComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.passed { "PASS" } else { "FAIL" };
        write!(
            f,
            "[{status}] spectral: LSD={lsd:.3}dB, SC={sc:.6}, max_mag_diff={max_mag:.3}dB, \
             mean_mag_diff={mean_mag:.3}dB, phase_coh={phase:.4}",
            lsd = self.log_spectral_distance_db,
            sc = self.spectral_convergence,
            max_mag = self.max_magnitude_diff_db,
            mean_mag = self.mean_magnitude_diff_db,
            phase = self.phase_coherence,
        )
    }
}

// ---------------------------------------------------------------------------
// Spectral comparison
// ---------------------------------------------------------------------------

/// Floor value for magnitude (dB) to avoid log(0).
const MAG_FLOOR: f32 = 1e-10;

/// Compare two 1-D audio signals in the spectral domain.
///
/// Computes STFT of both signals, then calculates log-spectral distance,
/// spectral convergence, per-bin magnitude difference, and phase coherence.
pub fn compare_spectral(
    reference: &[f32],
    candidate: &[f32],
    config: &SpectralConfig,
) -> Result<SpectralComparison, ReftestError> {
    if reference.is_empty() || candidate.is_empty() {
        return Err(ReftestError::EmptyTensor(
            "spectral comparison input".into(),
        ));
    }

    let (ref_mag, ref_phase) = stft_full(reference, &config.stft)?;
    let (cand_mag, cand_phase) = stft_full(candidate, &config.stft)?;

    // Both spectrograms must have same shape (guaranteed if signals are same
    // length, but handle gracefully otherwise).
    if ref_mag.n_freqs != cand_mag.n_freqs || ref_mag.n_frames != cand_mag.n_frames {
        return Err(ReftestError::SpectralConfig(format!(
            "spectrogram shape mismatch: ref [{}, {}] vs cand [{}, {}]",
            ref_mag.n_freqs, ref_mag.n_frames, cand_mag.n_freqs, cand_mag.n_frames,
        )));
    }

    let total_bins = ref_mag.data.len();
    if total_bins == 0 {
        return Err(ReftestError::EmptyTensor(
            "spectral comparison produced 0 bins".into(),
        ));
    }

    // --- Log-spectral distance ---
    // LSD = sqrt(mean( (10*log10(S_ref) - 10*log10(S_cand))^2 ))
    let mut sum_sq_log_diff: f64 = 0.0;

    // --- Spectral convergence ---
    // SC = ||S_ref - S_cand||_F / ||S_ref||_F
    let mut sum_sq_diff: f64 = 0.0;
    let mut sum_sq_ref: f64 = 0.0;

    // --- Per-bin magnitude difference (dB) ---
    let mut max_mag_diff_db: f32 = 0.0;
    let mut sum_mag_diff_db: f64 = 0.0;

    // --- Phase coherence ---
    let mut sum_cos_phase_diff: f64 = 0.0;

    for i in 0..total_bins {
        let r = ref_mag.data[i].max(MAG_FLOOR);
        let c = cand_mag.data[i].max(MAG_FLOOR);

        // Log-spectral distance (power spectrum in dB).
        let r_db = 10.0 * f64::from(r).log10();
        let c_db = 10.0 * f64::from(c).log10();
        let db_diff = r_db - c_db;
        sum_sq_log_diff += db_diff * db_diff;

        // Spectral convergence (magnitude domain).
        let mag_diff = f64::from(r) - f64::from(c);
        sum_sq_diff += mag_diff * mag_diff;
        sum_sq_ref += f64::from(r) * f64::from(r);

        // Per-bin magnitude difference in dB.
        let abs_db_diff = db_diff.abs() as f32;
        if abs_db_diff > max_mag_diff_db {
            max_mag_diff_db = abs_db_diff;
        }
        sum_mag_diff_db += f64::from(abs_db_diff);

        // Phase coherence: cos(phase_ref - phase_cand).
        let phase_diff = ref_phase.data[i] - cand_phase.data[i];
        sum_cos_phase_diff += f64::from(phase_diff.cos());
    }

    let n = total_bins as f64;

    let log_spectral_distance_db = (sum_sq_log_diff / n).sqrt() as f32;

    let spectral_convergence = if sum_sq_ref > 0.0 {
        (sum_sq_diff.sqrt() / sum_sq_ref.sqrt()) as f32
    } else {
        // Reference is silence — any non-zero candidate is divergent.
        if sum_sq_diff > 0.0 {
            f32::INFINITY
        } else {
            0.0
        }
    };

    let mean_magnitude_diff_db = (sum_mag_diff_db / n) as f32;
    let phase_coherence = (sum_cos_phase_diff / n) as f32;

    let passed = log_spectral_distance_db <= config.max_lsd_db
        && spectral_convergence <= config.max_spectral_convergence
        && phase_coherence >= config.min_phase_coherence;

    Ok(SpectralComparison {
        log_spectral_distance_db,
        spectral_convergence,
        max_magnitude_diff_db: max_mag_diff_db,
        mean_magnitude_diff_db,
        phase_coherence,
        passed,
    })
}

#[cfg(test)]
#[path = "spectral_tests.rs"]
mod tests;
