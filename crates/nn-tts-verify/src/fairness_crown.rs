// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN-verified fairness bounds — Layer 2 of the two-layer fairness architecture.
//!
//! Layer 1 (`fairness.rs`) answers: "Is the model fair on this test set?"
//! Layer 2 (this module) answers: "Is the model fair for ALL possible inputs
//! within each group's region?"
//!
//! The key insight: different demographic groups map to different regions of the
//! embedding space. Japanese text features occupy a different subspace than
//! English text features. If CROWN bounds are tighter for one group's region,
//! the model is more predictable (fairer) for that group. Wider bounds indicate
//! more quality variation — less reliable quality.
//!
//! References:
//! - Design doc: `designs/archive/2026-03-10-provably-fair-voice.md` Phase 2

use crate::error::{DspErrorKind, InvalidConfigKind, TtsVerifyError};
use crate::fairness::Group;
use nn_verify::{propagate_with_crown_fallback, BoundedTensor, GraphNetwork, VerifyError};
use ndarray::{ArrayD, IxDyn};

/// Input region for a specific group, defined by per-element embedding bounds.
///
/// Constructed from corpus statistics: run the model on a representative corpus
/// per language/group, capture intermediate embeddings, compute per-element
/// `[min, max]` across the corpus. This gives empirical input regions.
#[derive(Debug, Clone)]
pub struct GroupInputRegion {
    /// The demographic/linguistic group this region represents.
    pub group: Group,
    /// Per-element lower bounds on the input tensor.
    pub lower: Vec<f64>,
    /// Per-element upper bounds on the input tensor.
    pub upper: Vec<f64>,
}

/// Result of CROWN verification for one group's input region.
#[derive(Debug, Clone)]
pub struct GroupBoundsResult {
    /// The group whose input region was verified.
    pub group: Group,
    /// Output bound width (mean across output elements).
    pub mean_output_width: f64,
    /// Output bound width (max across output elements).
    pub max_output_width: f64,
    /// Per-output-element lower bounds.
    pub output_lower: Vec<f64>,
    /// Per-output-element upper bounds.
    pub output_upper: Vec<f64>,
    /// Propagation mode used (Crown or Ibp).
    pub propagation_mode: String,
}

/// CROWN-verified fairness certificate: compares output bound widths across groups.
///
/// Wider bounds = more quality variation = less reliable quality for that group.
/// If bound widths are approximately equal across groups, the model is formally fair.
#[derive(Debug, Clone)]
pub struct FairnessBoundsCertificate {
    /// Per-group verification results.
    pub group_results: Vec<GroupBoundsResult>,
    /// Maximum ratio of mean bound widths between any two groups.
    /// A ratio of 1.0 means perfectly equal bounds; 2.0 means one group has
    /// 2x the output variation.
    pub max_width_ratio: f64,
    /// Threshold for "fair" (default: 2.0 — no group has >2x bound width).
    pub width_ratio_threshold: f64,
    /// Is the model formally fair? True if `max_width_ratio < width_ratio_threshold`.
    pub is_fair: bool,
}

/// Compute CROWN bounds for each group's input region and compare.
///
/// For each `GroupInputRegion`, creates `BoundedTensor` input bounds, runs
/// `propagate_with_crown_fallback()` through the graph, and computes output
/// bound widths. Then compares all pairs of groups to find the maximum ratio
/// of output bound widths.
///
/// # Arguments
///
/// * `graph` — A NY `GraphNetwork` representing the model (or submodel)
///   to verify. Typically created via `tensor_kernel_to_graph()`.
/// * `regions` — Per-group input regions. Each region's `lower`/`upper` must have
///   the same length, matching the graph's variable input dimensionality.
/// * `input_shape` — Shape of the input tensor (e.g., `&[129, 4]` for STFT magnitudes).
///   Used to reshape the flat `lower`/`upper` vectors into ndarrays.
/// * `width_ratio_threshold` — Maximum acceptable ratio of bound widths between
///   any two groups. Default recommendation: 2.0.
///
/// # Errors
///
/// Validate fairness bound regions before CROWN propagation.
///
/// Checks: non-empty, length consistency, finiteness, and lower <= upper.
///
/// Returns `TtsVerifyError` if:
/// - `regions` is empty
/// - Any region has mismatched lower/upper lengths
/// - Any region's length doesn't match the product of `input_shape`
/// - CROWN/IBP propagation fails for any group
pub(crate) fn validate_fairness_regions(
    regions: &[GroupInputRegion],
    input_shape: &[usize],
) -> Result<(), TtsVerifyError> {
    if regions.is_empty() {
        return Err(TtsVerifyError::EmptyInput);
    }

    let expected_len: usize = input_shape.iter().product();

    for region in regions {
        if region.lower.len() != region.upper.len() {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::Constraint {
                    what: "group input region lower/upper lengths must match",
                },
            ));
        }
        if region.lower.len() != expected_len {
            return Err(TtsVerifyError::InvalidConfig(
                InvalidConfigKind::Constraint {
                    what: "group bounds length must match input shape",
                },
            ));
        }
        for (&lo, &hi) in region.lower.iter().zip(region.upper.iter()) {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(TtsVerifyError::InvalidConfig(
                    InvalidConfigKind::NonFinite {
                        param: "group input region bounds",
                    },
                ));
            }
            if lo > hi {
                return Err(TtsVerifyError::InvalidConfig(
                    InvalidConfigKind::RangeInverted {
                        param: "group input region bounds",
                    },
                ));
            }
        }
    }

    Ok(())
}

pub fn verify_fairness_bounds(
    graph: &GraphNetwork,
    regions: &[GroupInputRegion],
    input_shape: &[usize],
    width_ratio_threshold: f64,
) -> Result<FairnessBoundsCertificate, TtsVerifyError> {
    validate_fairness_regions(regions, input_shape)?;

    // Propagate bounds for each group
    let mut group_results = Vec::with_capacity(regions.len());

    for region in regions {
        let lower_f32: Vec<f32> = region.lower.iter().map(|&v| v as f32).collect();
        let upper_f32: Vec<f32> = region.upper.iter().map(|&v| v as f32).collect();

        let lower_arr = ArrayD::from_shape_vec(IxDyn(input_shape), lower_f32).map_err(|e| {
            TtsVerifyError::OperationFailed {
                context: "reshape lower bounds for fairness group",
                source: Box::new(e),
            }
        })?;
        let upper_arr = ArrayD::from_shape_vec(IxDyn(input_shape), upper_f32).map_err(|e| {
            TtsVerifyError::OperationFailed {
                context: "reshape upper bounds for fairness group",
                source: Box::new(e),
            }
        })?;

        let input_bounds = BoundedTensor::new(lower_arr, upper_arr).map_err(|e| {
            TtsVerifyError::OperationFailed {
                context: "create BoundedTensor for fairness group",
                source: Box::new(e),
            }
        })?;

        let (method, output_bounds, _fallback_reason) =
            propagate_with_crown_fallback(graph, &input_bounds).map_err(|e: VerifyError| {
                TtsVerifyError::OperationFailed {
                    context: "CROWN/IBP propagation for fairness group",
                    source: Box::new(e),
                }
            })?;

        let (out_lo, out_hi) = output_bounds.lower_upper();

        // Validate output bounds finiteness (defense-in-depth: NY may
        // produce NaN/Inf outputs for pathological networks or numerical issues)
        for (&lo, &hi) in out_lo.iter().zip(out_hi.iter()) {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(TtsVerifyError::Dsp(DspErrorKind::Computation {
                    what: "non-finite output bounds from CROWN propagation",
                }));
            }
        }

        // Compute per-element output widths
        let widths: Vec<f64> = out_lo
            .iter()
            .zip(out_hi.iter())
            .map(|(&lo, &hi)| f64::from(hi - lo))
            .collect();

        let mean_output_width = if widths.is_empty() {
            0.0
        } else {
            widths.iter().sum::<f64>() / widths.len() as f64
        };
        let max_output_width =
            crate::stats::fold_max_propagate_nan(widths.iter().copied(), 0.0_f64);

        group_results.push(GroupBoundsResult {
            group: region.group.clone(),
            mean_output_width,
            max_output_width,
            output_lower: out_lo.iter().map(|&v| f64::from(v)).collect(),
            output_upper: out_hi.iter().map(|&v| f64::from(v)).collect(),
            propagation_mode: format!("{method:?}"),
        });
    }

    // Compare all pairs: max(width_a / width_b) for mean widths
    let max_width_ratio = compute_max_width_ratio(&group_results);
    let is_fair = max_width_ratio < width_ratio_threshold;

    Ok(FairnessBoundsCertificate {
        group_results,
        max_width_ratio,
        width_ratio_threshold,
        is_fair,
    })
}

/// Compute the maximum ratio of mean output widths between any two groups.
///
/// Returns 1.0 if fewer than 2 groups (trivially fair).
/// Guards against division by zero: if any group has zero width, uses
/// `max_output_width` as fallback, or returns `f64::INFINITY` if both are zero.
fn compute_max_width_ratio(results: &[GroupBoundsResult]) -> f64 {
    if results.len() < 2 {
        return 1.0;
    }

    let mut max_ratio = 1.0_f64;

    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let wa = results[i].mean_output_width;
            let wb = results[j].mean_output_width;

            // Skip if both are zero (degenerate constant model)
            if wa == 0.0 && wb == 0.0 {
                continue;
            }

            // Compute ratio (larger / smaller)
            let ratio = if wa > wb {
                if wb > 0.0 {
                    wa / wb
                } else {
                    f64::INFINITY
                }
            } else if wa > 0.0 {
                wb / wa
            } else {
                f64::INFINITY
            };

            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
    }

    max_ratio
}

#[cfg(test)]
#[path = "fairness_crown_tests.rs"]
mod tests;
