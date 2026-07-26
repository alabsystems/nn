// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Probabilistic moonshot property checks via concentration inequality bridges.
//!
//! When deterministic CROWN bounds are too wide to prove a property, combine
//! the CROWN range with empirical Monte Carlo samples using concentration
//! inequalities for a 99% confidence probabilistic bound.
//!
//! Phase 1: Hoeffding (epsilon from CROWN range).
//! Phase 2: ConcentrationCertificate — Hoeffding + McDiarmid (Lipschitz-based),
//!   takes the tighter epsilon per dimension (#2882).
//! Phase 3: Distributional propagation through CROWN linear relaxation — tighter
//!   bounds when input distribution is known (uniform from CROWN bounds) (#2882).
//!
//! Part of #2882, Part of #2463, Part of #2218.

use ny_propagate::probabilistic::concentration::{
    estimate_lipschitz_from_network, hoeffding_bound, ConcentrationCertificate,
};
use ny_propagate::probabilistic::distributional::{
    propagate_distribution, AnalyticDistribution,
};
use ny_propagate::{LinearBounds, Network};
use nn_verify::BoundedTensor;
use ndarray::{ArrayD, IxDyn};

use crate::moonshot::{VerificationLevel, PROPERTY_NAMES};
use crate::pipeline::PipelineCertificate;

use super::MoonshotPropertyResult;

/// Check Property 2 (non-clipping) with probabilistic fallback.
///
/// First tries deterministic CROWN check. If bounds are too wide (output bounds
/// exceed [-1, 1]), falls back to concentration inequality bridge.
///
/// When `network` is provided, tries McDiarmid (Lipschitz-based) alongside
/// Hoeffding and takes the tighter epsilon per dimension. When `None`,
/// uses Hoeffding only (Phase 1 behavior).
pub fn check_non_clipping_probabilistic(
    cert: &PipelineCertificate,
    empirical_outputs: &ArrayD<f32>,
    num_samples: usize,
    confidence: f64,
    network: Option<&Network>,
) -> MoonshotPropertyResult {
    let det_result = super::check_non_clipping(cert);
    if det_result.proven {
        return det_result;
    }

    let crown_bounds = match bounded_tensor_from_cert(cert) {
        Some(bt) => bt,
        None => return det_result,
    };

    let (epsilons, method) = concentration_epsilons(
        empirical_outputs,
        &crown_bounds,
        cert,
        network,
        num_samples,
        confidence,
    );
    let epsilons = match epsilons {
        Some(e) => e,
        None => return det_result,
    };

    // Check: for each dimension, empirical_mean +/- epsilon must be in [-1, 1].
    let means: Vec<f64> = empirical_outputs.iter().map(|&x| f64::from(x)).collect();
    let all_within = means
        .iter()
        .zip(epsilons.iter())
        .all(|(&m, &eps)| m + eps <= 1.0 && m - eps >= -1.0);

    let worst_bound = means
        .iter()
        .zip(epsilons.iter())
        .map(|(&m, &eps)| (m + eps).abs().max((m - eps).abs()))
        .fold(0.0_f64, f64::max);

    let level = if all_within {
        VerificationLevel::CrownProbabilistic
    } else {
        VerificationLevel::Empirical
    };

    MoonshotPropertyResult {
        property_index: 1,
        property_name: PROPERTY_NAMES[1],
        proven: all_within,
        level,
        bound_value: worst_bound,
        threshold: 1.0,
        is_sound: cert.is_sound,
        explanation: format!(
            "{method} (n={num_samples}, conf={confidence:.2}): worst={worst_bound:.6}, \
             target [-1, 1]: {}",
            if all_within {
                "PROBABILISTIC"
            } else {
                "NOT PROVEN"
            }
        ),
    }
}

/// Check Property 1 (non-silence) with probabilistic fallback.
///
/// Falls back to concentration inequality bridge when deterministic CROWN bound
/// is too weak to prove that output is non-trivially non-zero.
///
/// When `network` is provided, tries McDiarmid alongside Hoeffding.
pub fn check_non_silence_probabilistic(
    cert: &PipelineCertificate,
    empirical_outputs: &ArrayD<f32>,
    num_samples: usize,
    confidence: f64,
    rms_threshold: f64,
    network: Option<&Network>,
) -> MoonshotPropertyResult {
    let det_result = super::check_non_silence(cert, rms_threshold);
    if det_result.proven {
        return det_result;
    }

    let crown_bounds = match bounded_tensor_from_cert(cert) {
        Some(bt) => bt,
        None => return det_result,
    };

    let (epsilons, method) = concentration_epsilons(
        empirical_outputs,
        &crown_bounds,
        cert,
        network,
        num_samples,
        confidence,
    );
    let epsilons = match epsilons {
        Some(e) => e,
        None => return det_result,
    };

    // Non-silence: at least one dimension has |mean| - epsilon > threshold.
    let means: Vec<f64> = empirical_outputs.iter().map(|&x| f64::from(x)).collect();
    let any_nonsilent = means
        .iter()
        .zip(epsilons.iter())
        .any(|(&m, &eps)| m.abs() - eps > rms_threshold);

    let best_bound = means
        .iter()
        .zip(epsilons.iter())
        .map(|(&m, &eps)| m.abs() - eps)
        .fold(f64::NEG_INFINITY, f64::max);

    let level = if any_nonsilent {
        VerificationLevel::CrownProbabilistic
    } else {
        VerificationLevel::Empirical
    };

    MoonshotPropertyResult {
        property_index: 0,
        property_name: PROPERTY_NAMES[0],
        proven: any_nonsilent,
        level,
        bound_value: best_bound.max(0.0),
        threshold: rms_threshold,
        is_sound: cert.is_sound,
        explanation: format!(
            "{method} (n={num_samples}, conf={confidence:.2}): best_abs_lower={best_bound:.6}, \
             threshold={rms_threshold:.6}: {}",
            if any_nonsilent {
                "PROBABILISTIC"
            } else {
                "NOT PROVEN"
            }
        ),
    }
}

/// Check Property 2 (non-clipping) using distributional propagation through CROWN.
///
/// Requires CROWN linear bounds (A_L, b_L, A_U, b_U) from the final pipeline stage.
/// Assumes uniform input distribution over the CROWN input range. Gives tighter
/// probabilistic bounds than Hoeffding when the final stage is linear/near-linear
/// (e.g., iSTFT: DFT matmul + Hann window + overlap-add — all linear ops).
///
/// Part of #2882 (D2) — Phase 3 distributional propagation.
pub fn check_non_clipping_distributional(
    linear_bounds: &LinearBounds,
    input_bounds: &BoundedTensor,
    confidence: f64,
    is_sound: bool,
) -> MoonshotPropertyResult {
    let dist = AnalyticDistribution::UniformFromBounds;
    let result = match propagate_distribution(linear_bounds, &dist, input_bounds, confidence) {
        Ok(r) => r,
        Err(_) => {
            return MoonshotPropertyResult {
                property_index: 1,
                property_name: PROPERTY_NAMES[1],
                proven: false,
                level: VerificationLevel::Empirical,
                bound_value: f64::INFINITY,
                threshold: 1.0,
                is_sound,
                explanation: "distributional propagation failed".into(),
            }
        }
    };

    // Check: probabilistic bounds [prob_lower_i, prob_upper_i] all within [-1, 1].
    let all_within = result
        .prob_lower
        .iter()
        .zip(result.prob_upper.iter())
        .all(|(&lo, &up)| lo >= -1.0 && up <= 1.0);

    let worst = result
        .prob_lower
        .iter()
        .zip(result.prob_upper.iter())
        .map(|(&lo, &up)| f64::from(lo.abs()).max(f64::from(up.abs())))
        .fold(0.0_f64, f64::max);

    let level = if all_within {
        VerificationLevel::CrownProbabilistic
    } else {
        VerificationLevel::Empirical
    };

    MoonshotPropertyResult {
        property_index: 1,
        property_name: PROPERTY_NAMES[1],
        proven: all_within,
        level,
        bound_value: worst,
        threshold: 1.0,
        is_sound,
        explanation: format!(
            "Distributional (conf={confidence:.2}): worst={worst:.6}, target [-1, 1]: {}",
            if all_within {
                "DISTRIBUTIONAL"
            } else {
                "NOT PROVEN"
            }
        ),
    }
}

/// Verify properties P1-P3, P6 with probabilistic fallback for each.
///
/// For each property, first tries deterministic CROWN. If that fails, applies
/// concentration inequality bridge using the provided empirical outputs.
///
/// When `network` is provided, McDiarmid (Lipschitz-based) bounds are tried
/// alongside Hoeffding, taking the tighter epsilon per dimension.
///
/// When `final_stage_linear` is provided, uses distributional propagation for P2
/// (non-clipping) through the CROWN linear relaxation of the final stage.
pub fn verify_properties_probabilistic(
    cert: &PipelineCertificate,
    empirical_outputs: &ArrayD<f32>,
    dim: usize,
    num_samples: usize,
    confidence: f64,
    network: Option<&Network>,
    final_stage_linear: Option<(&LinearBounds, &BoundedTensor)>,
) -> super::MoonshotCrownBundle {
    // P2 (non-clipping): prefer distributional path when linear bounds available,
    // fall back to concentration inequality bridge.
    let p2_result = if let Some((lb, ib)) = final_stage_linear {
        let dist = check_non_clipping_distributional(lb, ib, confidence, cert.is_sound);
        if dist.proven {
            dist
        } else {
            // Distributional didn't prove it — try concentration fallback.
            check_non_clipping_probabilistic(
                cert,
                empirical_outputs,
                num_samples,
                confidence,
                network,
            )
        }
    } else {
        check_non_clipping_probabilistic(cert, empirical_outputs, num_samples, confidence, network)
    };

    let results = vec![
        check_non_silence_probabilistic(
            cert,
            empirical_outputs,
            num_samples,
            confidence,
            0.01,
            network,
        ),
        p2_result,
        super::check_intelligibility_proxy(cert, 100.0),
        super::check_streaming_safety(cert, 240, 0.3),
    ];

    let all_proven = results.iter().all(|r| r.proven);

    super::MoonshotCrownBundle {
        results,
        pipeline_cert: cert.clone(),
        verification_dim: dim,
        all_proven,
    }
}

/// Compute per-dimension epsilon using ConcentrationCertificate.
///
/// When `network` is provided, builds a combined Hoeffding + McDiarmid certificate
/// and returns the tighter epsilon per dimension. Falls back to Hoeffding-only
/// when Lipschitz estimation fails (overflow for deep models, etc.).
///
/// Returns `(Some(epsilons), method_name)` on success, `(None, _)` on failure.
fn concentration_epsilons(
    empirical_outputs: &ArrayD<f32>,
    crown_bounds: &BoundedTensor,
    cert: &PipelineCertificate,
    network: Option<&Network>,
    num_samples: usize,
    confidence: f64,
) -> (Option<Vec<f64>>, &'static str) {
    // Try combined certificate when network is available
    if let Some(net) = network {
        if let Some(input_bounds) = input_bounded_tensor_from_cert(cert) {
            let lip = estimate_lipschitz_from_network(net);
            if let Ok(lip_est) = lip {
                // Skip McDiarmid if Lipschitz overflowed to infinity (#4145)
                if lip_est.value.is_finite() {
                    let combined = ConcentrationCertificate::compute_with_mcdiarmid_optimistic(
                        empirical_outputs,
                        crown_bounds,
                        empirical_outputs,
                        &input_bounds,
                        &lip_est,
                        num_samples,
                        confidence,
                        true, // bonferroni correction
                    );
                    if let Ok(cert) = combined {
                        let epsilons = tightest_epsilons(&cert);
                        let method = if cert.mcdiarmid_bounds.is_some() {
                            "Hoeffding+McDiarmid"
                        } else {
                            "Hoeffding"
                        };
                        return (Some(epsilons), method);
                    }
                }
            }
        }
    }

    // Fallback: Hoeffding only
    match hoeffding_bound(empirical_outputs, crown_bounds, num_samples, confidence) {
        Ok(bounds) => {
            let epsilons = bounds.iter().map(|hb| hb.epsilon).collect();
            (Some(epsilons), "Hoeffding")
        }
        Err(_) => (None, "Hoeffding"),
    }
}

/// Extract the tightest epsilon per dimension from a ConcentrationCertificate.
///
/// When McDiarmid bounds are available, takes `min(hoeffding.epsilon, mcdiarmid.epsilon)`
/// per dimension. Otherwise uses Hoeffding epsilon directly.
fn tightest_epsilons(cert: &ConcentrationCertificate) -> Vec<f64> {
    match &cert.mcdiarmid_bounds {
        Some(mcdiarmid) => cert
            .hoeffding_bounds
            .iter()
            .zip(mcdiarmid.iter())
            .map(|(h, m)| h.epsilon.min(m.epsilon))
            .collect(),
        None => cert.hoeffding_bounds.iter().map(|h| h.epsilon).collect(),
    }
}

/// Build a `BoundedTensor` from pipeline certificate's end-to-end OUTPUT bounds.
fn bounded_tensor_from_cert(cert: &PipelineCertificate) -> Option<BoundedTensor> {
    bounded_tensor_from_vecs(&cert.e2e_output_lower, &cert.e2e_output_upper)
}

/// Build a `BoundedTensor` from pipeline certificate's end-to-end INPUT bounds.
fn input_bounded_tensor_from_cert(cert: &PipelineCertificate) -> Option<BoundedTensor> {
    bounded_tensor_from_vecs(&cert.e2e_input_lower, &cert.e2e_input_upper)
}

/// Build a `BoundedTensor` from f64 lower/upper vectors.
fn bounded_tensor_from_vecs(lower: &[f64], upper: &[f64]) -> Option<BoundedTensor> {
    let lower_f32: Vec<f32> = lower.iter().map(|&x| x as f32).collect();
    let upper_f32: Vec<f32> = upper.iter().map(|&x| x as f32).collect();
    let n = lower_f32.len();
    if n == 0 {
        return None;
    }
    if !lower_f32
        .iter()
        .chain(upper_f32.iter())
        .all(|x| x.is_finite())
    {
        return None;
    }
    let lower_arr = ArrayD::from_shape_vec(IxDyn(&[n]), lower_f32).ok()?;
    let upper_arr = ArrayD::from_shape_vec(IxDyn(&[n]), upper_f32).ok()?;
    BoundedTensor::new(lower_arr, upper_arr).ok()
}
