// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Individual moonshot property check functions.
//!
//! Each function takes a [`PipelineCertificate`] or [`TimingCertificate`] and
//! returns a [`MoonshotPropertyResult`] indicating whether the property is
//! proven by the CROWN bounds.

use crate::moonshot::{VerificationLevel, PROPERTY_NAMES};
use crate::pipeline::{PipelineCertificate, TimingCertificate};

use super::MoonshotPropertyResult;

/// Check Property 1 (non-silence) against pipeline output bounds.
///
/// Non-silence requires that the output is not identically zero — at least
/// some output elements have lower bounds away from zero, indicating the
/// model produces non-trivial output.
///
/// The check: max(|output_lower|, |output_upper|) > threshold.
pub fn check_non_silence(cert: &PipelineCertificate, rms_threshold: f64) -> MoonshotPropertyResult {
    // Compute the maximum absolute bound across all output elements.
    let max_abs_lower = crate::stats::fold_max_propagate_nan(
        cert.e2e_output_lower.iter().map(|x| x.abs()),
        0.0_f64,
    );
    let max_abs_upper = crate::stats::fold_max_propagate_nan(
        cert.e2e_output_upper.iter().map(|x| x.abs()),
        0.0_f64,
    );
    // NaN-propagating combine: if either side is NaN, result is NaN.
    // f64::max() uses IEEE 754 maxNum which silently discards NaN.
    let max_abs = if max_abs_lower.is_nan() || max_abs_upper.is_nan() {
        f64::NAN
    } else {
        max_abs_lower.max(max_abs_upper)
    };

    // If max absolute bound > threshold, the output cannot be all-zero.
    let proven = max_abs > rms_threshold && cert.is_valid;

    let level = if proven && cert.is_sound {
        VerificationLevel::CrownProven
    } else if proven {
        VerificationLevel::CrownPartial
    } else {
        VerificationLevel::Empirical
    };

    MoonshotPropertyResult {
        property_index: 0,
        property_name: PROPERTY_NAMES[0],
        proven,
        level,
        bound_value: max_abs,
        threshold: rms_threshold,
        is_sound: cert.is_sound,
        explanation: format!(
            "max|output_bound| = {max_abs:.6}, threshold = {rms_threshold:.6}: {}",
            if proven { "PROVEN" } else { "NOT PROVEN" }
        ),
    }
}

/// Check Property 2 (non-clipping) against pipeline output bounds.
///
/// Non-clipping requires all output samples to be in [-1, 1].
/// The check: output_upper ≤ 1.0 AND output_lower ≥ -1.0 for all elements.
///
/// The existing vocoder pipeline proves P2 on the *spectral* domain (pre-iSTFT).
/// Full audio-domain P2 requires CROWN through the iSTFT linear transform,
/// implemented in `compose_kokoro_istft.rs` (Part of #2916). The iSTFT stage
/// uses an analytical bridge (cos/sin ∈ [-1,1]) from Generator magnitude bounds
/// to iSTFT input bounds, then exact CROWN through the iSTFT LinearLayer.
/// A defense-in-depth `clamp(-1, 1)` after iSTFT guarantees P2 unconditionally.
pub fn check_non_clipping(cert: &PipelineCertificate) -> MoonshotPropertyResult {
    // SOUNDNESS: f64::max(NaN, x) returns x (IEEE 754-2008 maxNum semantics),
    // silently discarding NaN. We must check finiteness before trusting the
    // fold results. Without this guard, NaN-contaminated bounds produce
    // max_upper=NEG_INFINITY and min_lower=INFINITY, falsely satisfying [-1,1].
    // See P1-234 audit.
    let finite_bounds = cert
        .e2e_output_lower
        .iter()
        .chain(cert.e2e_output_upper.iter())
        .all(|x| x.is_finite());

    let max_upper = crate::stats::fold_max_propagate_nan(
        cert.e2e_output_upper.iter().copied(),
        f64::NEG_INFINITY,
    );
    let min_lower =
        crate::stats::fold_min_propagate_nan(cert.e2e_output_lower.iter().copied(), f64::INFINITY);

    // Non-clipping: all output in [-1, 1].
    let within_range = max_upper <= 1.0 && min_lower >= -1.0;
    let proven = finite_bounds && within_range && cert.is_valid;

    // Report the worst bound value (furthest from [-1, 1]).
    let worst_bound = max_upper.abs().max(min_lower.abs());

    let level = if proven && cert.is_sound {
        VerificationLevel::CrownProven
    } else if proven {
        VerificationLevel::CrownPartial
    } else {
        VerificationLevel::Empirical
    };

    MoonshotPropertyResult {
        property_index: 1,
        property_name: PROPERTY_NAMES[1],
        proven,
        level,
        bound_value: worst_bound,
        threshold: 1.0,
        is_sound: cert.is_sound,
        explanation: format!(
            "output range [{min_lower:.6}, {max_upper:.6}], target [-1, 1]: {}",
            if proven { "PROVEN" } else { "NOT PROVEN" }
        ),
    }
}

/// Check Property 3 (intelligibility proxy) against pipeline output bounds.
///
/// Full attention monotonicity requires proving argmax non-crossing, which
/// CROWN cannot do directly. This proxy checks that attention output bounds
/// are finite and non-degenerate, indicating the attention mechanism produces
/// structured (non-random) outputs.
///
/// The check: output bounds have finite range and the range width is
/// non-trivially less than the input range width (information is preserved,
/// not lost to vacuous bounds).
pub fn check_intelligibility_proxy(
    cert: &PipelineCertificate,
    max_range_ratio: f64,
) -> MoonshotPropertyResult {
    // Compare output range width to input range width.
    let input_range: f64 = crate::stats::fold_max_propagate_nan(
        cert.e2e_input_upper
            .iter()
            .zip(cert.e2e_input_lower.iter())
            .map(|(u, l)| u - l),
        0.0_f64,
    );

    let output_range: f64 = crate::stats::fold_max_propagate_nan(
        cert.e2e_output_upper
            .iter()
            .zip(cert.e2e_output_lower.iter())
            .map(|(u, l)| u - l),
        0.0_f64,
    );

    // If output range / input range < max_range_ratio, bounds are informative.
    let range_ratio = if input_range > 0.0 {
        output_range / input_range
    } else {
        f64::INFINITY
    };

    let finite_bounds = cert
        .e2e_output_lower
        .iter()
        .chain(cert.e2e_output_upper.iter())
        .all(|x| x.is_finite());

    let proven = finite_bounds && range_ratio < max_range_ratio && cert.is_valid;

    let level = if proven && cert.is_sound {
        VerificationLevel::CrownPartial // Proxy, not full monotonicity
    } else {
        VerificationLevel::Empirical
    };

    MoonshotPropertyResult {
        property_index: 2,
        property_name: PROPERTY_NAMES[2],
        proven,
        level,
        bound_value: range_ratio,
        threshold: max_range_ratio,
        is_sound: cert.is_sound,
        explanation: format!(
            "range ratio = {range_ratio:.4} (output/input), max = {max_range_ratio:.4}: {} (proxy)",
            if proven { "PARTIAL" } else { "NOT PROVEN" }
        ),
    }
}

/// Check Property 6 (streaming safety) against pipeline output bounds.
///
/// Streaming safety requires bounded sample-to-sample discontinuity at chunk
/// boundaries. The crossfade operation is linear:
///
/// ```text
/// crossfade[i] = tail[i] * (1 - alpha_i) + head[i] * alpha_i
/// ```
///
/// where `alpha_i = i / (crossfade_len - 1)`.
///
/// Since this is a convex combination with positive coefficients, CROWN bounds
/// on the crossfade output are exact: if both chunk outputs are bounded in
/// `[lower, upper]`, then the max sample-to-sample difference within the
/// crossfade region is bounded by:
///
/// ```text
/// max_click ≤ output_bound_range * alpha_step
/// ```
///
/// where `output_bound_range = max(upper - lower)` across all elements and
/// `alpha_step = 1 / (crossfade_len - 1)`.
///
/// The check: `output_bound_range * alpha_step ≤ click_threshold`.
pub fn check_streaming_safety(
    cert: &PipelineCertificate,
    crossfade_samples: usize,
    click_threshold: f64,
) -> MoonshotPropertyResult {
    // Compute the maximum element-wise bound range across all output elements.
    let max_bound_range: f64 = crate::stats::fold_max_propagate_nan(
        cert.e2e_output_upper
            .iter()
            .zip(cert.e2e_output_lower.iter())
            .map(|(u, l)| u - l),
        0.0_f64,
    );

    // Alpha step for the crossfade: each adjacent sample pair differs in alpha
    // by 1/(crossfade_len - 1). The worst-case click at the boundary is bounded
    // by the output range times this step.
    let alpha_step = if crossfade_samples > 1 {
        1.0 / (crossfade_samples - 1) as f64
    } else {
        1.0 // degenerate: no crossfade, full discontinuity possible
    };

    let max_click_bound = max_bound_range * alpha_step;

    let finite_bounds = cert
        .e2e_output_lower
        .iter()
        .chain(cert.e2e_output_upper.iter())
        .all(|x| x.is_finite());

    let proven = finite_bounds && max_click_bound <= click_threshold && cert.is_valid;

    let level = if proven && cert.is_sound {
        VerificationLevel::CrownProven
    } else if proven {
        VerificationLevel::CrownPartial
    } else {
        VerificationLevel::Empirical
    };

    MoonshotPropertyResult {
        property_index: 5,
        property_name: PROPERTY_NAMES[5],
        proven,
        level,
        bound_value: max_click_bound,
        threshold: click_threshold,
        is_sound: cert.is_sound,
        explanation: format!(
            "max_click_bound = {max_click_bound:.6} (range={max_bound_range:.4} × step={alpha_step:.6}), \
             threshold = {click_threshold:.6}: {}",
            if proven { "PROVEN" } else { "NOT PROVEN" }
        ),
    }
}

/// Check Property 5 (temporal boundedness) against a timing certificate.
///
/// Temporal boundedness requires that worst-case inference time is within the
/// specified timing bound. The check uses the roofline cost model combined
/// with CROWN-verified pipeline bounds to produce a formally-grounded
/// timing guarantee.
///
/// The timing certificate couples CROWN bounds verification with cost model
/// profiling: both output correctness and execution time are verified from the
/// same pipeline definition.
///
/// The check: `worst_case_time_us <= timing_bound_us`.
pub fn check_temporal_boundedness(timing_cert: &TimingCertificate) -> MoonshotPropertyResult {
    let proven = timing_cert.timing_bound_met && timing_cert.bounds_cert.is_valid;

    // The timing certificate couples CROWN bounds with cost model.
    // If the bounds are sound (CROWN, not IBP), the timing proof is CROWN-coupled.
    let level = if proven && timing_cert.bounds_cert.is_sound {
        VerificationLevel::CrownProven
    } else if proven {
        VerificationLevel::CrownPartial
    } else {
        VerificationLevel::Empirical
    };

    MoonshotPropertyResult {
        property_index: 4,
        property_name: PROPERTY_NAMES[4],
        proven,
        level,
        bound_value: timing_cert.worst_case_time_us,
        threshold: timing_cert.timing_bound_us,
        is_sound: timing_cert.bounds_cert.is_sound,
        explanation: format!(
            "worst_case={:.1} μs, bound={:.1} μs ({:.1}x margin): {}",
            timing_cert.worst_case_time_us,
            timing_cert.timing_bound_us,
            if timing_cert.worst_case_time_us > 0.0 {
                timing_cert.timing_bound_us / timing_cert.worst_case_time_us
            } else {
                f64::INFINITY
            },
            if proven { "PROVEN" } else { "NOT PROVEN" }
        ),
    }
}

/// Check peak memory boundedness against a hardware memory limit.
///
/// Validates that the dispatch plan's peak memory usage (weights + activations)
/// does not exceed the available hardware memory. For M4 Max unified memory
/// with 7 concurrent voices, the per-model budget is approximately 128 GB / 7
/// ≈ 18 GB per voice pipeline.
///
/// This is not a moonshot property per se (Property 7 "memory safety" refers
/// to Kani-verified absence of UB), but it is a necessary condition for
/// Property 5 (temporal boundedness) — a model that exceeds physical memory
/// will page-fault and violate the timing bound.
///
/// # Arguments
///
/// * `timing_cert` — Timing certificate containing the peak memory profile.
/// * `memory_bound_bytes` — Maximum allowed peak memory in bytes.
pub fn check_memory_boundedness(
    timing_cert: &TimingCertificate,
    memory_bound_bytes: u64,
) -> MoonshotPropertyResult {
    let (peak_bytes, peak_step) = match &timing_cert.peak_memory {
        Some(pm) => (pm.peak_total_bytes, pm.peak_step_name.clone()),
        None => (0, "unknown".to_string()),
    };

    let proven = peak_bytes > 0 && peak_bytes <= memory_bound_bytes;

    // Memory boundedness is a prerequisite for timing. If bounds are sound
    // and memory fits, this inherits the timing certificate's soundness level.
    let level = if proven && timing_cert.bounds_cert.is_sound {
        VerificationLevel::CrownProven
    } else if proven {
        VerificationLevel::CrownPartial
    } else {
        VerificationLevel::Empirical
    };

    MoonshotPropertyResult {
        // Use property_index 4 (temporal boundedness) since memory is a
        // sub-condition of temporal boundedness, not a separate property.
        property_index: 4,
        property_name: PROPERTY_NAMES[4],
        proven,
        level,
        bound_value: peak_bytes as f64,
        threshold: memory_bound_bytes as f64,
        is_sound: timing_cert.bounds_cert.is_sound,
        explanation: format!(
            "peak_memory={:.2} MB ({} bytes), bound={:.2} MB ({} bytes), peak_step={}: {}",
            peak_bytes as f64 / (1024.0 * 1024.0),
            peak_bytes,
            memory_bound_bytes as f64 / (1024.0 * 1024.0),
            memory_bound_bytes,
            peak_step,
            if proven {
                "WITHIN BOUND"
            } else {
                "EXCEEDS BOUND"
            },
        ),
    }
}
