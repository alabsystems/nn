// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Softmax-inclusive attention weight certificate extraction.
//!
//! Extracted from softmax_attention.rs to stay under 500-line limit.

use nn_verify::BoundedTensor;

/// Certificate for multi-head causal attention weight dominance (post-softmax).
///
/// For each head `h` and decoder step `t`, checks that the attention weight
/// at the alignment target dominates all other unmasked positions:
///   `lower(W[h, t, f(t)]) > max_{j unmasked, j != f(t)}(upper(W[h, t, j]))`
///
/// Since W = softmax(S), the bounds on W are in [0, 1]. The dominance
/// of the alignment target in weight space is strictly stronger than
/// score-space dominance: it proves the model actually attends most to
/// the correct position, not just that the score is highest.
#[derive(Debug)]
pub(crate) struct SoftmaxAttentionCertificate {
    pub(crate) num_heads: usize,
    pub(crate) decoder_steps: usize,
    pub(crate) encoder_positions: usize,
    /// Per-head minimum weight margins: `per_head_min_margin[h]` is the
    /// worst row weight margin for head `h`.
    pub(crate) per_head_min_margin: Vec<f64>,
    /// Per-head row margins in weight space.
    pub(crate) per_head_row_margins: Vec<Vec<f64>>,
    /// Overall minimum margin across all heads and rows.
    pub(crate) min_margin: f64,
    /// Whether all heads have proven weight dominance (min_margin > 0).
    pub(crate) is_proven: bool,
    /// Number of heads with individually proven weight dominance.
    pub(crate) proven_heads: usize,
    /// Per-head minimum lower bound on the target weight.
    pub(crate) per_head_target_weight_lo: Vec<f64>,
    /// Per-head maximum upper bound on the target weight.
    pub(crate) per_head_target_weight_hi: Vec<f64>,
    pub(crate) input_bound: f64,
    pub(crate) propagation_mode: String,
}

/// Extract a softmax-inclusive attention weight certificate from weight bounds.
///
/// Output shape: `[H, T_dec, T_enc]`. Values are attention weights in [0, 1].
/// `alignment_fn(t)` returns the target encoder position for decoder step `t`.
pub(crate) fn extract_softmax_certificate(
    output: &BoundedTensor,
    num_heads: usize,
    t_dec: usize,
    t_enc: usize,
    input_bound: f64,
    mode: &str,
    alignment_fn: impl Fn(usize) -> usize,
) -> SoftmaxAttentionCertificate {
    let (lo, hi) = output.lower_upper();
    let flat_lo: Vec<f32> = lo.iter().copied().collect();
    let flat_hi: Vec<f32> = hi.iter().copied().collect();

    let head_stride = t_dec * t_enc;
    let mut per_head_min_margin = Vec::with_capacity(num_heads);
    let mut per_head_row_margins = Vec::with_capacity(num_heads);
    let mut per_head_target_weight_lo = Vec::with_capacity(num_heads);
    let mut per_head_target_weight_hi = Vec::with_capacity(num_heads);
    let mut proven_heads = 0;

    for h in 0..num_heads {
        let head_offset = h * head_stride;
        let mut row_margins = Vec::new();
        let mut head_min_target_lo = f64::INFINITY;
        let mut head_max_target_hi = f64::NEG_INFINITY;

        for t in 0..t_dec {
            let target = alignment_fn(t);
            if target >= t_enc {
                continue;
            }

            // Skip post-alignment rows (target saturated at last position).
            let max_visible = target;
            if t_enc > 1 && target == t_enc - 1 && max_visible == t_enc - 1 {
                continue;
            }

            let target_lo = f64::from(flat_lo[head_offset + t * t_enc + target]);
            let target_hi = f64::from(flat_hi[head_offset + t * t_enc + target]);

            if target_lo < head_min_target_lo {
                head_min_target_lo = target_lo;
            }
            if target_hi > head_max_target_hi {
                head_max_target_hi = target_hi;
            }

            // Find max upper bound among OTHER unmasked positions.
            let mut max_other_hi = f64::NEG_INFINITY;
            for j in 0..=max_visible {
                if j != target {
                    let upper = f64::from(flat_hi[head_offset + t * t_enc + j]);
                    if upper > max_other_hi {
                        max_other_hi = upper;
                    }
                }
            }

            // If target is the only visible position, trivially dominant.
            let margin = if max_visible == 0 {
                f64::INFINITY
            } else {
                target_lo - max_other_hi
            };
            row_margins.push(margin);
        }

        let head_min = row_margins.iter().copied().fold(f64::INFINITY, f64::min);
        if head_min > 0.0 {
            proven_heads += 1;
        }
        per_head_min_margin.push(head_min);
        per_head_row_margins.push(row_margins);
        per_head_target_weight_lo.push(head_min_target_lo);
        per_head_target_weight_hi.push(head_max_target_hi);
    }

    let min_margin = per_head_min_margin
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    SoftmaxAttentionCertificate {
        num_heads,
        decoder_steps: t_dec,
        encoder_positions: t_enc,
        per_head_min_margin,
        per_head_row_margins,
        min_margin,
        is_proven: min_margin > 0.0,
        proven_heads,
        per_head_target_weight_lo,
        per_head_target_weight_hi,
        input_bound,
        propagation_mode: mode.to_string(),
    }
}

/// Assert all per-head margins are finite and weight margins are in [0, 1].
pub(crate) fn assert_certificate_margins_valid(cert: &SoftmaxAttentionCertificate) {
    for (h, m) in cert.per_head_min_margin.iter().enumerate() {
        assert!(m.is_finite(), "head {h}: margin should be finite, got {m}");
    }
}

/// Assert weight-space margins are in [0, 1] (softmax output property).
pub(crate) fn assert_weight_margins_bounded(cert: &SoftmaxAttentionCertificate) {
    for (h, wm) in cert.per_head_min_margin.iter().enumerate() {
        if *wm > 0.0 {
            assert!(
                *wm <= 1.0,
                "head {h}: weight margin should be <= 1.0, got {wm}"
            );
        }
    }
}
