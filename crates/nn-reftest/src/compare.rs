// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tensor comparison engine: per-layer divergence detection.

use crate::error::ReftestError;
use crate::trace::{NamedTensor, ReferenceTrace};

// Configuration types and result structs (ComparisonConfig, LayerComparison,
// DivergenceReport) extracted to stay under the 500-line limit (Part of #1575).
#[path = "compare_config.rs"]
mod config;
pub use config::{ComparisonConfig, DivergenceReport, LayerComparison};

/// Compare two named tensors element-wise.
///
/// Returns a `LayerComparison` with max/mean absolute difference, RMS difference,
/// cosine similarity, max relative difference, and peak amplitude. The `passed`
/// field reflects whether all metrics are within the given `config` tolerances.
#[must_use = "returns a Result containing the comparison"]
pub fn compare_tensors(
    reference: &NamedTensor,
    candidate: &NamedTensor,
    config: &ComparisonConfig,
) -> Result<LayerComparison, ReftestError> {
    if reference.shape != candidate.shape {
        return Err(ReftestError::ShapeMismatch {
            name: reference.name.clone(),
            expected: reference.shape.clone(),
            actual: candidate.shape.clone(),
        });
    }

    let n = reference.data.len();
    if n == 0 {
        return Err(ReftestError::EmptyTensor(reference.name.clone()));
    }

    let ref_data = &reference.data;
    let cand_data = &candidate.data;

    // Compute all metrics in a single pass.
    // IEEE 754: NaN/Inf elements are detected inline and treated as infinite
    // divergence. Without explicit is_finite() checks, NaN comparisons silently
    // return false, causing max_abs to stay at 0.0. See design doc.
    let mut max_abs: f32 = 0.0;
    let mut sum_abs: f64 = 0.0;
    let mut sum_sq_diff: f64 = 0.0;
    let mut max_rel: f32 = 0.0;
    let mut dot: f64 = 0.0;
    let mut norm_ref: f64 = 0.0;
    let mut norm_cand: f64 = 0.0;
    let mut peak_amp: f32 = 0.0;
    let mut has_non_finite = false;

    for i in 0..n {
        let r = ref_data[i];
        let c = cand_data[i];

        // Track peak amplitude of candidate regardless of finiteness.
        // Non-finite candidates get INFINITY peak (caught by amplitude gate).
        if !c.is_finite() {
            peak_amp = f32::INFINITY;
        } else {
            let c_abs = c.abs();
            if c_abs > peak_amp {
                peak_amp = c_abs;
            }
        }

        // Treat NaN/Inf elements as maximum divergence.
        if !r.is_finite() || !c.is_finite() {
            max_abs = f32::INFINITY;
            sum_abs = f64::INFINITY;
            sum_sq_diff = f64::INFINITY;
            max_rel = f32::INFINITY;
            has_non_finite = true;
            continue;
        }

        let abs_diff = (r - c).abs();

        if abs_diff > max_abs {
            max_abs = abs_diff;
        }
        let diff64 = f64::from(abs_diff);
        sum_abs += diff64;
        sum_sq_diff += diff64 * diff64;

        // Skip relative error for near-zero values where absolute error
        // already confirms correctness. When both values are smaller than
        // abs_tolerance, even tiny absolute differences produce misleadingly
        // large relative errors (e.g., 1e-8 diff on 3e-7 values = 3% rel).
        if r.abs() >= config.abs_tolerance || c.abs() >= config.abs_tolerance {
            let denom = r.abs().max(c.abs()).max(1e-8);
            let rel = abs_diff / denom;
            if rel > max_rel {
                max_rel = rel;
            }
        }

        let r64 = f64::from(r);
        let c64 = f64::from(c);
        dot += r64 * c64;
        norm_ref += r64 * r64;
        norm_cand += c64 * c64;
    }

    let mean_abs = (sum_abs / n as f64) as f32;
    let rms_diff = (sum_sq_diff / n as f64).sqrt() as f32;

    let cosine_similarity = if has_non_finite && norm_ref == 0.0 && norm_cand == 0.0 {
        // All elements were non-finite — no finite data to compute cosine over.
        // Report NaN to signal "undefined" rather than a misleading 1.0.
        f32::NAN
    } else if norm_ref == 0.0 && norm_cand == 0.0 {
        // Both zero vectors (all finite): treat as identical.
        1.0_f32
    } else if norm_ref == 0.0 || norm_cand == 0.0 {
        // One zero, one non-zero: no meaningful similarity.
        0.0_f32
    } else {
        (dot / (norm_ref.sqrt() * norm_cand.sqrt())) as f32
    };

    let mut passed = max_abs <= config.abs_tolerance
        && max_rel <= config.rel_tolerance
        && cosine_similarity >= config.cosine_threshold;

    // Apply optional gates.
    if let Some(rms_limit) = config.rms_tolerance {
        if rms_diff > rms_limit {
            passed = false;
        }
    }
    if let Some(peak_limit) = config.peak_amplitude_limit {
        if peak_amp > peak_limit {
            passed = false;
        }
    }

    // Spectral comparison for 1-D audio tensors.
    #[cfg(feature = "spectral")]
    let (spectral_result, passed) = {
        if let Some(ref spectral_config) = config.spectral {
            // Only run spectral comparison on 1-D tensors (waveforms).
            if reference.shape.len() == 1 {
                match crate::spectral::compare_spectral(ref_data, cand_data, spectral_config) {
                    Ok(sc) => {
                        let spectral_passed = sc.passed;
                        (Some(sc), passed && spectral_passed)
                    }
                    Err(_) => (None, passed),
                }
            } else {
                (None, passed)
            }
        } else {
            (None, passed)
        }
    };

    Ok(LayerComparison {
        name: reference.name.clone(),
        shape: reference.shape.clone(),
        max_abs_diff: max_abs,
        mean_abs_diff: mean_abs,
        cosine_similarity,
        max_rel_diff: max_rel,
        num_elements: n,
        rms_diff,
        peak_amplitude: peak_amp,
        passed,
        #[cfg(feature = "spectral")]
        spectral: spectral_result,
    })
}

/// Compare two traces layer-by-layer.
///
/// Traces are matched by position (index). Both traces must have the same
/// number of checkpoints. For name-based matching, pre-filter or reorder
/// traces before calling this function.
#[must_use = "returns a Result containing the divergence report"]
pub fn compare_traces(
    reference: &ReferenceTrace,
    candidate: &ReferenceTrace,
    config: &ComparisonConfig,
) -> Result<DivergenceReport, ReftestError> {
    if reference.len() != candidate.len() {
        return Err(ReftestError::TraceLengthMismatch {
            reference: reference.len(),
            candidate: candidate.len(),
        });
    }

    let mut layers = Vec::with_capacity(reference.len());
    let mut first_failure: Option<usize> = None;

    for (i, (ref_tensor, cand_tensor)) in reference.iter().zip(candidate.iter()).enumerate() {
        let comparison = compare_tensors(ref_tensor, cand_tensor, config)?;
        if !comparison.passed && first_failure.is_none() {
            first_failure = Some(i);
        }
        layers.push(comparison);
    }

    let all_passed = first_failure.is_none();

    Ok(DivergenceReport {
        layers,
        first_failure,
        all_passed,
    })
}

#[cfg(test)]
#[path = "compare_tests.rs"]
mod tests;
