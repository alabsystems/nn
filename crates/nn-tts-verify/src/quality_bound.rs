// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN-based audio quality bound certificates for TTS.
//!
//! Bridges formal CROWN output bounds to quality metric guarantees
//! via Lipschitz-continuity arguments. The key insight:
//!
//! 1. CROWN proves: for any input in perturbation set, output changes by at
//!    most δ (measured by the output bound width).
//! 2. Quality metrics (SNR, spectral convergence, MCD) are Lipschitz-continuous
//!    with known constants L.
//! 3. Therefore: quality metric change ≤ L × δ (worst case).
//! 4. If baseline_quality - L × δ ≥ threshold, quality is formally guaranteed.
//!
//! This gives the first formal quality guarantees for TTS under adversarial
//! perturbations — not by running CROWN through the quality metric itself
//! (which is non-graph-representable), but by composing CROWN output bounds
//! with analytical metric properties.
//!
//! # Lipschitz Constants
//!
//! - **SNR**: `L_snr = 20 / (ln(10) × σ_noise)` dB per unit output perturbation,
//!   where σ_noise is the baseline noise RMS (derived from signal RMS and SNR).
//! - **Spectral convergence**: `L_sc = 1 / ‖S_ref‖_F` per unit perturbation
//!   in Frobenius norm on the magnitude spectrum.
//! - **MCD**: `L_mcd ≈ (10√2 / ln(10)) × (1 / n_frames)` dB per unit perturbation,
//!   bounded via DCT isometry (Kubichek 1993).
//! - **Cosine similarity**: `L_cos = 1 / ‖x‖₂` per unit perturbation.
//!
//! Part of #1740: Adversarial Robustness of TTS — AC3.
//!
//! # References
//!
//! - Kubichek (1993). "Mel-cepstral distance measure." IEEE ICASSP.
//! - Taal et al. (2011). "STOI." IEEE TASLP.
//! - ITU-T P.862.2 (2005). PESQ-MOS.

use crate::error::{DspErrorKind, TtsVerifyError};

/// A quality metric with its Lipschitz constant for perturbation analysis.
#[derive(Debug, Clone)]
pub struct QualityMetricSpec {
    /// Metric name (e.g., "SNR", "spectral_convergence", "MCD").
    pub name: String,
    /// Lipschitz constant: max quality change per unit output perturbation.
    pub lipschitz_constant: f64,
    /// Baseline quality value at unperturbed input.
    pub baseline_value: f64,
    /// Minimum acceptable quality value (threshold).
    pub threshold: f64,
    /// Whether higher is better (true for SNR, STOI) or lower is better (true for MCD).
    pub higher_is_better: bool,
    /// Academic citation for the Lipschitz bound.
    pub citation: &'static str,
}

/// Result of quality bound verification for one metric.
#[derive(Debug, Clone)]
pub struct QualityBoundResult {
    /// Metric name.
    pub metric_name: String,
    /// CROWN-proven output bound width (δ).
    pub output_bound_width: f64,
    /// Lipschitz constant used.
    pub lipschitz_constant: f64,
    /// Maximum possible quality degradation: L × δ.
    pub max_quality_change: f64,
    /// Baseline quality at unperturbed input.
    pub baseline_value: f64,
    /// Worst-case quality: baseline ± L×δ (direction depends on higher_is_better).
    pub worst_case_value: f64,
    /// Pass/fail threshold.
    pub threshold: f64,
    /// Whether quality is guaranteed to remain above threshold.
    pub guaranteed: bool,
    /// Margin: worst_case_value distance from threshold (positive = safe).
    pub margin: f64,
}

/// Certificate proving audio quality bounds hold under adversarial perturbations.
#[derive(Debug, Clone)]
pub struct QualityBoundCertificate {
    /// Per-metric results.
    pub metric_results: Vec<QualityBoundResult>,
    /// Whether ALL metrics are guaranteed to hold.
    pub all_guaranteed: bool,
    /// The metric with the smallest margin (closest to failing).
    pub tightest_metric: String,
    /// Smallest margin across all metrics.
    pub tightest_margin: f64,
    /// CROWN output bound width used for all metrics.
    pub output_bound_width: f64,
}

/// Compute quality bound certificates from CROWN output bounds.
///
/// Given a CROWN-proven output bound width (δ) and a set of quality metrics
/// with Lipschitz constants, compute the worst-case quality for each metric
/// and determine whether quality is formally guaranteed.
///
/// # Arguments
///
/// * `output_bound_width` — CROWN-proven mean output bound width (δ).
/// * `metrics` — Quality metrics with Lipschitz constants and baselines.
///
/// # Returns
///
/// A `QualityBoundCertificate` with per-metric guarantees.
pub fn verify_quality_bounds(
    output_bound_width: f64,
    metrics: &[QualityMetricSpec],
) -> Result<QualityBoundCertificate, TtsVerifyError> {
    if !output_bound_width.is_finite() || output_bound_width < 0.0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "output_bound_width must be finite and non-negative",
        }));
    }
    if metrics.is_empty() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::EmptyInput {
            what: "metrics list is empty",
        }));
    }

    let mut results = Vec::with_capacity(metrics.len());

    for spec in metrics {
        validate_metric_spec(spec)?;

        let max_quality_change = spec.lipschitz_constant * output_bound_width;

        // Worst-case quality: degrade from baseline in the bad direction.
        let worst_case = if spec.higher_is_better {
            spec.baseline_value - max_quality_change
        } else {
            spec.baseline_value + max_quality_change
        };

        // Margin: distance from threshold in the safe direction.
        let (guaranteed, margin) = if spec.higher_is_better {
            // Higher is better: worst_case >= threshold is safe.
            let m = worst_case - spec.threshold;
            (m >= 0.0, m)
        } else {
            // Lower is better: worst_case <= threshold is safe.
            let m = spec.threshold - worst_case;
            (m >= 0.0, m)
        };

        results.push(QualityBoundResult {
            metric_name: spec.name.clone(),
            output_bound_width,
            lipschitz_constant: spec.lipschitz_constant,
            max_quality_change,
            baseline_value: spec.baseline_value,
            worst_case_value: worst_case,
            threshold: spec.threshold,
            guaranteed,
            margin,
        });
    }

    let all_guaranteed = results.iter().all(|r| r.guaranteed);
    let tightest = results
        .iter()
        .min_by(|a, b| {
            a.margin
                .partial_cmp(&b.margin)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| (r.metric_name.clone(), r.margin))
        .unwrap_or_else(|| (String::new(), 0.0));

    Ok(QualityBoundCertificate {
        metric_results: results,
        all_guaranteed,
        tightest_metric: tightest.0,
        tightest_margin: tightest.1,
        output_bound_width,
    })
}

/// Validate a metric spec for finiteness and consistency.
fn validate_metric_spec(spec: &QualityMetricSpec) -> Result<(), TtsVerifyError> {
    if !spec.lipschitz_constant.is_finite() || spec.lipschitz_constant < 0.0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "lipschitz_constant must be finite and non-negative",
        }));
    }
    if !spec.baseline_value.is_finite() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "baseline_value must be finite",
        }));
    }
    if !spec.threshold.is_finite() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "threshold must be finite",
        }));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Lipschitz constant computation helpers
// ---------------------------------------------------------------------------

/// Compute SNR Lipschitz constant from signal RMS and baseline SNR.
///
/// SNR = 20 × log10(‖signal‖ / ‖noise‖). For output perturbation δ,
/// the noise increases by at most δ (triangle inequality):
///
///   ΔSNR ≤ 20 / (ln(10) × noise_rms) × δ
///
/// where noise_rms = signal_rms × 10^(-baseline_snr_db / 20), derived
/// from the baseline SNR measurement.
///
/// The derivative of `20 × log10(S / N)` w.r.t. noise magnitude N is
/// `20 / (ln(10) × N)`. This is evaluated at the baseline noise level,
/// giving a sound upper bound (the derivative decreases as N increases,
/// so the operating-point value is the tightest valid Lipschitz constant).
///
/// # Soundness note
///
/// Previous versions used `signal_rms` as the denominator instead of
/// `noise_rms`. This was unsound: for 25 dB SNR, noise is ~18× smaller
/// than signal, so the Lipschitz constant was underestimated by ~18×.
pub fn snr_lipschitz(signal_rms: f64, baseline_snr_db: f64) -> Result<f64, TtsVerifyError> {
    if !signal_rms.is_finite() || signal_rms <= 0.0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "signal_rms must be positive and finite",
        }));
    }
    if !baseline_snr_db.is_finite() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "baseline_snr_db must be finite",
        }));
    }
    // Derive baseline noise: SNR = 20*log10(S/N) → N = S * 10^(-SNR/20)
    let noise_rms = signal_rms * 10.0_f64.powf(-baseline_snr_db / 20.0);
    if !noise_rms.is_finite() || noise_rms <= 0.0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::Computation {
            what: "derived noise_rms must be positive and finite",
        }));
    }
    Ok(20.0 / (10.0_f64.ln() * noise_rms))
}

/// Compute spectral convergence Lipschitz constant from reference energy.
///
/// Spectral convergence = ‖S_ref - S_cand‖_F / ‖S_ref‖_F. Perturbation δ
/// in the time domain maps to at most δ in spectral domain (Parseval's theorem),
/// so: Δ(SC) ≤ δ / ‖S_ref‖_F.
///
/// # Domain assumption
///
/// Parseval's theorem gives exact energy preservation for the DFT per frame,
/// but STFT with overlap-add and windowing can amplify or attenuate the
/// effective norm by a factor depending on hop/window ratio. This constant
/// assumes the STFT is approximately norm-preserving (valid when using
/// normalized windows with hop_length ≈ window_length/4).
pub fn spectral_convergence_lipschitz(
    reference_spectral_energy: f64,
) -> Result<f64, TtsVerifyError> {
    if !reference_spectral_energy.is_finite() || reference_spectral_energy <= 0.0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "reference_spectral_energy must be positive and finite",
        }));
    }
    Ok(1.0 / reference_spectral_energy)
}

/// Compute MCD Lipschitz constant from frame count and MFCC parameters.
///
/// MCD = (10√2/ln10) × (1/T) × Σ_t ‖c_t - r_t‖₂. The DCT is an isometry
/// (preserves L2 norm), so spectral perturbation maps 1:1 to MFCC perturbation.
/// Per-frame MCD change ≤ (10√2/ln10) × δ_frame, and averaging over T frames
/// gives: Δ(MCD) ≤ (10√2/ln10) × δ / √T.
///
/// The √T factor comes from distributing a total L2 perturbation δ across T
/// frames (each frame gets at most δ/√T in the worst case by Cauchy-Schwarz).
///
/// # Domain assumption
///
/// This constant assumes δ is measured in MFCC space (post-DCT). The DCT
/// step is isometric, but the full audio→STFT→mel→log→DCT pipeline is not:
/// the log-mel step has Lipschitz constant ≈ 1/min(mel_energy), which can
/// be large for quiet frames. If δ is the CROWN output bound in audio space,
/// the effective Lipschitz constant through the full chain is larger.
pub fn mcd_lipschitz(n_frames: usize) -> Result<f64, TtsVerifyError> {
    if n_frames == 0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "n_frames must be positive for MCD Lipschitz",
        }));
    }
    let scale = 10.0 * 2.0_f64.sqrt() / 10.0_f64.ln();
    Ok(scale / (n_frames as f64).sqrt())
}

/// Compute cosine similarity Lipschitz constant from signal norm.
///
/// cos(x, y) = x·y / (‖x‖‖y‖). For perturbation δ to x:
/// Δ(cos) ≤ δ / ‖x‖₂ (worst case when perturbation is orthogonal to x).
pub fn cosine_similarity_lipschitz(signal_l2_norm: f64) -> Result<f64, TtsVerifyError> {
    if !signal_l2_norm.is_finite() || signal_l2_norm <= 0.0 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
            param: "signal_l2_norm must be positive and finite",
        }));
    }
    Ok(1.0 / signal_l2_norm)
}

/// Build a standard set of quality metric specs for TTS verification.
///
/// Given measured signal statistics from the unperturbed reference output,
/// constructs `QualityMetricSpec`s with analytically-derived Lipschitz
/// constants for SNR, spectral convergence, MCD, and cosine similarity.
///
/// # Arguments
///
/// * `signal_rms` — RMS of the unperturbed output signal.
/// * `signal_l2_norm` — L2 norm of the unperturbed output signal.
/// * `reference_spectral_energy` — Frobenius norm of the reference STFT magnitude.
/// * `n_frames` — Number of MFCC frames in the signal.
/// * `baseline_snr` — Measured SNR of the unperturbed output (dB).
/// * `baseline_sc` — Measured spectral convergence (0 = identical).
/// * `baseline_mcd` — Measured MCD of the unperturbed output (dB).
/// * `baseline_cosine` — Measured cosine similarity of the unperturbed output.
pub fn standard_quality_specs(
    signal_rms: f64,
    signal_l2_norm: f64,
    reference_spectral_energy: f64,
    n_frames: usize,
    baseline_snr: f64,
    baseline_sc: f64,
    baseline_mcd: f64,
    baseline_cosine: f64,
) -> Result<Vec<QualityMetricSpec>, TtsVerifyError> {
    Ok(vec![
        QualityMetricSpec {
            name: "SNR".into(),
            lipschitz_constant: snr_lipschitz(signal_rms, baseline_snr)?,
            baseline_value: baseline_snr,
            threshold: 10.0, // Minimum 10 dB SNR for intelligible speech
            higher_is_better: true,
            citation: "ITU-T P.56 (2011). Objective measurement of active speech level.",
        },
        QualityMetricSpec {
            name: "spectral_convergence".into(),
            lipschitz_constant: spectral_convergence_lipschitz(reference_spectral_energy)?,
            baseline_value: baseline_sc,
            threshold: 0.5, // Maximum 0.5 spectral convergence (lower is better)
            higher_is_better: false,
            citation: "Arik et al. (2018). Neural Voice Cloning. ICLR.",
        },
        QualityMetricSpec {
            name: "MCD".into(),
            lipschitz_constant: mcd_lipschitz(n_frames)?,
            baseline_value: baseline_mcd,
            threshold: 6.0, // Maximum 6.0 dB MCD (lower is better)
            higher_is_better: false,
            citation: "Kubichek (1993). Mel-cepstral distance. IEEE ICASSP.",
        },
        QualityMetricSpec {
            name: "cosine_similarity".into(),
            lipschitz_constant: cosine_similarity_lipschitz(signal_l2_norm)?,
            baseline_value: baseline_cosine,
            threshold: 0.8, // Minimum 0.8 cosine similarity
            higher_is_better: true,
            citation: "Jia et al. (2018). Transfer Learning from Speaker Verification. NeurIPS.",
        },
    ])
}

#[cfg(test)]
#[path = "quality_bound_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "quality_bound_kani.rs"]
mod kani_proofs;
