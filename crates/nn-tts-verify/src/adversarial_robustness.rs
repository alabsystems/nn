// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CROWN-based adversarial robustness verification for TTS.
//!
//! Given a NY `GraphNetwork` and phoneme confusion sets, proves that
//! for ANY token substitution within a confusion set, the output change is bounded.
//!
//! The key contribution: token-set embedding bounds. Instead of bounding over
//! the full vocabulary (178 tokens, maximally conservative), we bound over
//! linguistically meaningful confusion sets (2-4 tokens, tight bounds).
//!
//! Part of #1740: Adversarial Robustness of TTS.
//!
//! # References
//!
//! Miller, G.A. & Nicely, P.E. (1955). "An Analysis of Perceptual Confusions
//! Among Some English Consonants." JASA.

use crate::adversarial::{embedding_bounds_for_token_set, ConfusionSet};
use crate::error::{DspErrorKind, TtsVerifyError};
use nn_verify::{propagate_with_crown_fallback, BoundedTensor, GraphNetwork};
use ndarray::{ArrayD, IxDyn};

/// What property to verify under perturbation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RobustnessProperty {
    /// Duration predictions remain positive (lower bound > 0).
    DurationPositive,
    /// F0 stays within [min_hz, max_hz].
    F0Bounded { min_hz: f64, max_hz: f64 },
    /// Output bound width stays below threshold (output doesn't change "too much").
    OutputStable { max_width: f64 },
}

/// Configuration for adversarial robustness verification.
#[derive(Debug, Clone)]
pub struct RobustnessConfig {
    /// Maximum number of positions to perturb simultaneously. Default: 1.
    pub max_perturbation_positions: usize,
    /// Which confusion sets to use.
    pub confusion_sets: Vec<ConfusionSet>,
    /// Output property to verify (e.g., "duration_positive", "f0_bounded").
    pub property: RobustnessProperty,
}

/// Robustness result for one perturbation position.
#[derive(Debug, Clone)]
pub struct PositionRobustness {
    /// Position index in the phoneme sequence.
    pub position: usize,
    /// Base token ID at this position.
    pub base_token: u32,
    /// Name of the confusion set applied.
    pub confusion_set: String,
    /// Output bound width when this position is perturbed.
    pub output_width: f64,
    /// Does the specified property hold under perturbation?
    pub property_holds: bool,
    /// Propagation mode used ("CROWN" or "IBP").
    pub propagation_mode: String,
}

/// Result of adversarial robustness verification.
#[derive(Debug, Clone)]
pub struct RobustnessCertificate {
    /// Per-position results.
    pub position_results: Vec<PositionRobustness>,
    /// Overall: is the model robust to all tested perturbations?
    pub is_robust: bool,
    /// Worst-case output bound width across all perturbation positions.
    pub worst_case_width: f64,
    /// Which position produced the worst case.
    pub worst_position: usize,
    /// Which confusion set produced the worst case.
    pub worst_confusion_set: String,
}

/// Verify adversarial robustness of a model for a given phoneme sequence.
///
/// For each position `p` in `base_tokens`:
/// 1. Find the confusion set containing `base_tokens[p]`
/// 2. Create embedding bounds with only position `p` perturbed (others fixed)
/// 3. Run CROWN propagation through the graph
/// 4. Measure output bound width and check if the property holds
///
/// # Arguments
///
/// * `graph` — NY GraphNetwork (typically PlBert encoder or duration predictor)
/// * `embedding_weights` — flattened `[vocab_size, embed_dim]` embedding table
/// * `vocab_size` — number of tokens in vocabulary
/// * `embed_dim` — embedding dimension per token
/// * `base_tokens` — reference phoneme sequence to verify robustness around
/// * `config` — robustness verification configuration
pub fn verify_robustness(
    graph: &GraphNetwork,
    embedding_weights: &[f64],
    vocab_size: usize,
    embed_dim: usize,
    base_tokens: &[u32],
    config: &RobustnessConfig,
) -> Result<RobustnessCertificate, TtsVerifyError> {
    if base_tokens.is_empty() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::EmptyInput {
            what: "base_tokens is empty",
        }));
    }
    if embedding_weights.len() != vocab_size * embed_dim {
        return Err(TtsVerifyError::DimensionMismatch {
            expected: vocab_size * embed_dim,
            actual: embedding_weights.len(),
            context: "embedding_weights length must equal vocab_size * embed_dim",
        });
    }

    let seq_len = base_tokens.len();
    let total_dim = seq_len * embed_dim;
    let mut position_results = Vec::new();

    // Validate all tokens up-front (point-bounds construction accesses
    // embedding_weights at any token's offset, not just perturbed ones).
    for &token in base_tokens.iter() {
        if (token as usize) >= vocab_size {
            return Err(TtsVerifyError::Dsp(DspErrorKind::InvalidParam {
                param: "base_tokens element exceeds vocab_size",
            }));
        }
    }

    // Test each position individually.
    for pos in 0..seq_len {
        let token = base_tokens[pos];

        // Find the confusion set containing this token.
        let cs = config
            .confusion_sets
            .iter()
            .find(|cs| cs.token_ids.contains(&token));

        let cs = match cs {
            Some(c) => c,
            None => continue, // Token not in any confusion set — skip.
        };

        // Build embedding bounds: all positions fixed except `pos`.
        let mut lower = Vec::with_capacity(total_dim);
        let mut upper = Vec::with_capacity(total_dim);

        for (p, &tok) in base_tokens.iter().enumerate() {
            if p == pos {
                // Perturbed position: bounds span the confusion set.
                let (lo, hi) = embedding_bounds_for_token_set(
                    embedding_weights,
                    vocab_size,
                    embed_dim,
                    &cs.token_ids,
                )?;
                lower.extend_from_slice(&lo);
                upper.extend_from_slice(&hi);
            } else {
                // Fixed position: point bounds.
                let offset = (tok as usize) * embed_dim;
                for d in 0..embed_dim {
                    let val = embedding_weights[offset + d];
                    lower.push(val);
                    upper.push(val);
                }
            }
        }

        // Convert to f32 for BoundedTensor.
        let lower_f32: Vec<f32> = lower.iter().map(|&v| v as f32).collect();
        let upper_f32: Vec<f32> = upper.iter().map(|&v| v as f32).collect();

        let lower_arr =
            ArrayD::from_shape_vec(IxDyn(&[seq_len, embed_dim]), lower_f32).map_err(|e| {
                TtsVerifyError::OperationFailed {
                    context: "failed to build lower bounds array",
                    source: Box::new(e),
                }
            })?;
        let upper_arr =
            ArrayD::from_shape_vec(IxDyn(&[seq_len, embed_dim]), upper_f32).map_err(|e| {
                TtsVerifyError::OperationFailed {
                    context: "failed to build upper bounds array",
                    source: Box::new(e),
                }
            })?;

        let input_bounds = BoundedTensor::new(lower_arr, upper_arr).map_err(|e| {
            TtsVerifyError::OperationFailed {
                context: "failed to create BoundedTensor",
                source: Box::new(e),
            }
        })?;

        // Propagate through the graph.
        let (method, output, _fallback) = propagate_with_crown_fallback(graph, &input_bounds)
            .map_err(|e| TtsVerifyError::OperationFailed {
                context: "CROWN propagation failed",
                source: Box::new(e),
            })?;

        // Measure output bound width.
        let (lo_out, hi_out) = output.lower_upper();
        let lo_slice = lo_out.as_slice().ok_or_else(|| {
            TtsVerifyError::Dsp(DspErrorKind::Computation {
                what: "output lower bounds not contiguous",
            })
        })?;
        let hi_slice = hi_out.as_slice().ok_or_else(|| {
            TtsVerifyError::Dsp(DspErrorKind::Computation {
                what: "output upper bounds not contiguous",
            })
        })?;

        let output_width = compute_mean_width(lo_slice, hi_slice)?;
        let property_holds = check_property(&config.property, lo_slice, hi_slice)?;

        position_results.push(PositionRobustness {
            position: pos,
            base_token: token,
            confusion_set: cs.name.clone(),
            output_width,
            property_holds,
            propagation_mode: format!("{method:?}"),
        });
    }

    // Find worst case.
    let (worst_pos, worst_width, worst_cs) =
        position_results
            .iter()
            .fold((0, 0.0_f64, String::new()), |(wp, ww, wcs), r| {
                if r.output_width > ww {
                    (r.position, r.output_width, r.confusion_set.clone())
                } else {
                    (wp, ww, wcs)
                }
            });

    let is_robust = position_results.iter().all(|r| r.property_holds);

    Ok(RobustnessCertificate {
        position_results,
        is_robust,
        worst_case_width: worst_width,
        worst_position: worst_pos,
        worst_confusion_set: worst_cs,
    })
}

/// Compute mean bound width across all output dimensions.
pub(crate) fn compute_mean_width(lower: &[f32], upper: &[f32]) -> Result<f64, TtsVerifyError> {
    if lower.len() != upper.len() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::SizeMismatch {
            what: "lower/upper bounds length mismatch",
            expected: lower.len(),
            got: upper.len(),
        }));
    }
    if lower.is_empty() {
        return Ok(0.0);
    }
    let mut total = 0.0_f64;
    for (lo, hi) in lower.iter().zip(upper.iter()) {
        let w = f64::from(*hi) - f64::from(*lo);
        if !w.is_finite() {
            return Err(TtsVerifyError::Dsp(DspErrorKind::Computation {
                what: "non-finite bound width detected",
            }));
        }
        total += w;
    }
    Ok(total / lower.len() as f64)
}

/// Check whether the robustness property holds given output bounds.
pub(crate) fn check_property(
    property: &RobustnessProperty,
    lower: &[f32],
    upper: &[f32],
) -> Result<bool, TtsVerifyError> {
    match property {
        RobustnessProperty::DurationPositive => {
            // All output lower bounds must be > 0.
            Ok(lower.iter().all(|&v| v > 0.0))
        }
        RobustnessProperty::F0Bounded { min_hz, max_hz } => {
            // All outputs must be within [min_hz, max_hz].
            let min_f32 = *min_hz as f32;
            let max_f32 = *max_hz as f32;
            Ok(lower.iter().all(|&v| v >= min_f32) && upper.iter().all(|&v| v <= max_f32))
        }
        RobustnessProperty::OutputStable { max_width } => {
            // Mean output bound width must be below threshold.
            let width = compute_mean_width(lower, upper)?;
            Ok(width <= *max_width)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mean_width() {
        let lo = [0.0f32, 1.0, 2.0];
        let hi = [1.0f32, 3.0, 4.0];
        let w = compute_mean_width(&lo, &hi).unwrap();
        // widths: 1, 2, 2 → mean = 5/3
        assert!((w - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_mean_width_empty() {
        let w = compute_mean_width(&[], &[]).unwrap();
        assert_eq!(w, 0.0);
    }

    #[test]
    fn test_check_property_duration_positive() {
        let lo = [0.1f32, 0.2, 0.05];
        let hi = [1.0f32, 2.0, 0.5];
        assert!(check_property(&RobustnessProperty::DurationPositive, &lo, &hi).unwrap());

        let lo_neg = [-0.1f32, 0.2, 0.05];
        assert!(!check_property(&RobustnessProperty::DurationPositive, &lo_neg, &hi).unwrap());
    }

    #[test]
    fn test_check_property_f0_bounded() {
        let prop = RobustnessProperty::F0Bounded {
            min_hz: 80.0,
            max_hz: 400.0,
        };
        let lo = [85.0f32, 120.0];
        let hi = [350.0f32, 380.0];
        assert!(check_property(&prop, &lo, &hi).unwrap());

        let lo_low = [50.0f32, 120.0];
        assert!(!check_property(&prop, &lo_low, &hi).unwrap());
    }

    #[test]
    fn test_check_property_output_stable() {
        let prop = RobustnessProperty::OutputStable { max_width: 1.0 };
        let lo = [0.0f32, 0.0];
        let hi = [0.5f32, 0.5];
        assert!(check_property(&prop, &lo, &hi).unwrap());

        let hi_wide = [2.0f32, 2.0];
        assert!(!check_property(&prop, &lo, &hi_wide).unwrap());
    }
}
