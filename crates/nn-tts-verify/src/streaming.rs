// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Streaming voice verification — chunk boundary quality proofs.
//!
//! Verifies three properties at each boundary between adjacent audio chunks:
//!
//! - **P1 (No-Click):** Sample-to-sample discontinuity below threshold.
//! - **P2 (Crossfade Energy):** Energy in the crossfade region within bounds
//!   relative to chunk interior energy (no dips or spikes).
//! - **P3 (Spectral Continuity):** Spectral convergence at the boundary below
//!   threshold (no frequency cancellation from phase mismatch).
//!
//! Designed for Kokoro TTS's 40ms / 960-sample crossfade at 24kHz between
//! streaming chunks.

use crate::dsp;
use crate::error::{validate_finite_positive, DspErrorKind, InvalidConfigKind, TtsVerifyError};
use crate::multi_res_stft;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Configuration for streaming boundary verification.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StreamingConfig {
    /// Sample rate in Hz. Must match synthesis rate.
    pub sample_rate: u32,
    /// Crossfade length in samples (default: 960 = 40ms at 24kHz).
    pub crossfade_samples: usize,
    /// Boundary margin in samples for analysis (default: 1920 = 80ms at 24kHz).
    /// The margin defines the region around each boundary to inspect.
    /// Must be >= crossfade_samples.
    pub margin_samples: usize,
    /// Maximum sample-to-sample jump at boundary (P1). Default: 0.3.
    pub click_threshold: f64,
    /// Minimum crossfade energy as fraction of nominal (P2 low). Default: 0.5.
    pub energy_lo: f64,
    /// Maximum crossfade energy as fraction of nominal (P2 high). Default: 1.5.
    pub energy_hi: f64,
    /// Maximum spectral convergence at boundary (P3). Default: 0.15.
    pub spectral_threshold: f64,
}

impl StreamingConfig {
    /// Validate all configuration fields.
    pub fn validate(&self) -> Result<(), TtsVerifyError> {
        if self.sample_rate == 0 {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::Constraint {
                    what: "sample_rate must be > 0",
                },
            ));
        }
        if self.crossfade_samples == 0 {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::Constraint {
                    what: "crossfade_samples must be > 0",
                },
            ));
        }
        if self.margin_samples < self.crossfade_samples {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::Constraint {
                    what: "margin_samples must be >= crossfade_samples",
                },
            ));
        }
        validate_finite_positive(self.click_threshold, "click_threshold")?;
        validate_finite_positive(self.energy_lo, "energy_lo")?;
        validate_finite_positive(self.energy_hi, "energy_hi")?;
        validate_finite_positive(self.spectral_threshold, "spectral_threshold")?;
        if self.energy_lo >= self.energy_hi {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::RangeInverted {
                    param: "energy_lo/energy_hi",
                },
            ));
        }
        Ok(())
    }
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            crossfade_samples: 960, // 40ms at 24kHz — matches KokoroStreamConfig
            margin_samples: 1920,   // 80ms at 24kHz
            click_threshold: 0.3,
            energy_lo: 0.5,
            energy_hi: 1.5,
            spectral_threshold: 0.15,
        }
    }
}

/// Result of verifying a single chunk boundary.
#[derive(Debug, Clone)]
pub struct BoundaryResult {
    /// Index of the boundary (0 = between chunk 0 and chunk 1).
    pub boundary_index: usize,
    /// P1: maximum sample-to-sample difference in boundary region.
    pub max_click: f64,
    /// P2: RMS energy of crossfade region.
    pub crossfade_energy: f64,
    /// P2: nominal energy (average of chunk interiors).
    pub nominal_energy: f64,
    /// P2: energy ratio (crossfade_energy / nominal_energy).
    pub energy_ratio: f64,
    /// P3: spectral convergence at boundary.
    pub spectral_convergence: f64,
    /// Whether all three properties passed.
    pub passed: bool,
}

/// Aggregate streaming verification certificate.
#[derive(Debug, Clone)]
pub struct StreamingCertificate {
    /// Per-boundary results.
    pub boundaries: Vec<BoundaryResult>,
    /// Number of chunks analyzed.
    pub n_chunks: usize,
    /// Number of boundaries that passed all checks.
    pub n_passed: usize,
    /// Whether ALL boundaries passed (overall streaming quality).
    pub overall_passed: bool,
    /// Configuration used.
    pub config: StreamingConfig,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Verify streaming quality across chunk boundaries.
///
/// `chunks`: slice of PCM audio chunks from streaming synthesis.
/// Each chunk is a `&[f32]` slice of raw PCM samples.
///
/// Returns a [`StreamingCertificate`] with per-boundary results.
///
/// # Errors
///
/// Returns [`TtsVerifyError::Dsp`] if fewer than 2 chunks or any chunk is
/// shorter than `margin_samples`.
pub fn verify_streaming(
    chunks: &[&[f32]],
    config: &StreamingConfig,
) -> Result<StreamingCertificate, TtsVerifyError> {
    config.validate()?;

    if chunks.len() < 2 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InsufficientSamples {
            operation: "streaming verification",
            needed: 2,
            got: chunks.len(),
        }));
    }

    let mut boundaries = Vec::with_capacity(chunks.len() - 1);
    for i in 0..chunks.len() - 1 {
        let result = verify_boundary(chunks[i], chunks[i + 1], i, config)?;
        boundaries.push(result);
    }

    let n_passed = boundaries.iter().filter(|b| b.passed).count();
    let overall_passed = n_passed == boundaries.len();

    Ok(StreamingCertificate {
        n_chunks: chunks.len(),
        n_passed,
        overall_passed,
        boundaries,
        config: config.clone(),
    })
}

/// Verify a single boundary between two adjacent chunks.
///
/// `chunk_a`: the preceding chunk's PCM samples.
/// `chunk_b`: the following chunk's PCM samples.
/// `boundary_index`: index for reporting.
///
/// Extracts boundary regions, computes P1/P2/P3, and returns result.
pub fn verify_boundary(
    chunk_a: &[f32],
    chunk_b: &[f32],
    boundary_index: usize,
    config: &StreamingConfig,
) -> Result<BoundaryResult, TtsVerifyError> {
    config.validate()?;

    let m = config.margin_samples;
    let c = config.crossfade_samples;

    // Validate chunk lengths.
    if chunk_a.len() < m || chunk_b.len() < m {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InsufficientSamples {
            operation: "streaming boundary verification",
            needed: m,
            got: chunk_a.len().min(chunk_b.len()),
        }));
    }

    // Extract boundary regions.
    let tail = &chunk_a[chunk_a.len() - m..]; // last m samples of chunk_a
    let head = &chunk_b[..m]; // first m samples of chunk_b

    // P1: Click detection at the exact boundary.
    let boundary_sample_a = chunk_a[chunk_a.len() - 1];
    let boundary_sample_b = chunk_b[0];
    let raw_click = (f64::from(boundary_sample_b) - f64::from(boundary_sample_a)).abs();

    // Also check max_sample_diff in the margin region around boundary.
    let mut boundary_region = Vec::with_capacity(2 * m);
    boundary_region.extend_from_slice(tail);
    boundary_region.extend_from_slice(head);
    let max_click = dsp::max_sample_diff(&boundary_region).max(raw_click);

    // P2: Crossfade energy.
    let crossfade_tail = &chunk_a[chunk_a.len() - c..];
    let crossfade_head = &chunk_b[..c];
    let blended = crossfade_linear(crossfade_tail, crossfade_head)?;
    let crossfade_energy = dsp::rms(&blended);

    // Nominal energy: interior regions (exclude margin from each end).
    let interior_a = if chunk_a.len() > 2 * m {
        &chunk_a[m..chunk_a.len() - m]
    } else {
        &chunk_a[..chunk_a.len() / 2] // fallback for short chunks
    };
    let interior_b = if chunk_b.len() > 2 * m {
        &chunk_b[m..chunk_b.len() - m]
    } else {
        &chunk_b[chunk_b.len() / 2..] // fallback for short chunks
    };
    let energy_a = dsp::rms(interior_a);
    let energy_b = dsp::rms(interior_b);
    let nominal_energy = f64::midpoint(energy_a, energy_b);
    let energy_ratio = if nominal_energy > 1e-10 {
        crossfade_energy / nominal_energy
    } else {
        1.0 // both silent — ratio is nominal
    };

    // P3: Spectral convergence at boundary.
    //
    // Compare the boundary region's spectral content against the average
    // of the two chunk interiors. Uses smallest FFT size (512) for
    // boundary-local analysis.
    let spectral_convergence = if boundary_region.len() >= 512 {
        compute_boundary_spectral_convergence(&boundary_region, interior_a, interior_b)?
    } else {
        0.0 // too short for spectral analysis — pass vacuously
    };

    let passed = max_click <= config.click_threshold
        && energy_ratio >= config.energy_lo
        && energy_ratio <= config.energy_hi
        && spectral_convergence <= config.spectral_threshold;

    Ok(BoundaryResult {
        boundary_index,
        max_click,
        crossfade_energy,
        nominal_energy,
        energy_ratio,
        spectral_convergence,
        passed,
    })
}

/// Linear crossfade between two equal-length slices.
///
/// `tail[i] * (1 - alpha) + head[i] * alpha` where `alpha = i / (len - 1)`.
///
/// Delegates to [`nn_core::audio::crossfade_linear_blend`] for the core
/// blend math (shared with `nn-models` streaming assembler). The error
/// checking and `n <= 1` edge-case handling remain here to preserve the
/// public API contract.
///
/// This matches dvoice's crossfade implementation in `sentence.rs`.
///
/// # Errors
///
/// Returns [`TtsVerifyError::Dsp`] if `tail` and `head` have different lengths.
pub fn crossfade_linear(tail: &[f32], head: &[f32]) -> Result<Vec<f32>, TtsVerifyError> {
    if tail.len() != head.len() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::SizeMismatch {
            what: "crossfade_linear tail/head length",
            expected: tail.len(),
            got: head.len(),
        }));
    }
    let n = tail.len();
    if n <= 1 {
        // Preserve existing behavior: single-sample returns head, empty
        // returns empty. This differs from crossfade_linear_blend's
        // single-sample average -- kept for backward compatibility with
        // existing Kani proofs and tests.
        return Ok(head.to_vec());
    }
    Ok(nn_core::audio::crossfade_linear_blend(tail, head, n))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute spectral convergence of a boundary region relative to the average
/// spectral envelope of two adjacent chunk interiors.
///
/// Uses a single FFT size (512) for localized boundary analysis. Returns the
/// Frobenius-norm ratio `||boundary - reference||_F / ||reference||_F`.
fn compute_boundary_spectral_convergence(
    boundary_region: &[f32],
    interior_a: &[f32],
    interior_b: &[f32],
) -> Result<f64, TtsVerifyError> {
    let n_fft = 512;
    let hop = n_fft / 4;

    // Spectral magnitude of boundary region.
    let boundary_mag = multi_res_stft::stft_magnitude(boundary_region, n_fft, hop)?;

    // Spectral magnitude of interiors.
    let mag_a = multi_res_stft::stft_magnitude(interior_a, n_fft, hop)?;
    let mag_b = multi_res_stft::stft_magnitude(interior_b, n_fft, hop)?;

    // Average the interior magnitudes frame-wise.
    // Use the minimum frame count across all three spectrograms.
    let n_frames = boundary_mag.len().min(mag_a.len()).min(mag_b.len());
    if n_frames == 0 {
        return Ok(0.0);
    }

    let n_bins = boundary_mag[0].len();

    // Compute average interior spectrogram and Frobenius norm ratio.
    let mut diff_sq_sum = 0.0_f64;
    let mut ref_sq_sum = 0.0_f64;

    for t in 0..n_frames {
        let bound_frame = &boundary_mag[t];
        // Average of interior_a and interior_b for this frame index.
        // If one interior has fewer frames, clamp to last frame.
        let frame_a = &mag_a[t.min(mag_a.len() - 1)];
        let frame_b = &mag_b[t.min(mag_b.len() - 1)];

        for k in 0..n_bins.min(bound_frame.len()) {
            let ref_val = f64::midpoint(
                frame_a.get(k).copied().unwrap_or(0.0),
                frame_b.get(k).copied().unwrap_or(0.0),
            );
            let bound_val = bound_frame.get(k).copied().unwrap_or(0.0);
            let diff = bound_val - ref_val;
            diff_sq_sum += diff * diff;
            ref_sq_sum += ref_val * ref_val;
        }
    }

    if ref_sq_sum < 1e-20 {
        return Ok(0.0); // both reference and boundary are silent
    }

    Ok((diff_sq_sum / ref_sq_sum).sqrt())
}

#[cfg(test)]
#[path = "streaming_tests.rs"]
mod tests;
