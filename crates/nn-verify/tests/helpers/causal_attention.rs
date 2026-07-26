// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for causal cross-attention (decoder-side masking).
//!
//! In autoregressive TTS decoders, each decoder step `t` can only attend
//! to encoder positions `<= f(t)` where `f` is a monotonically increasing
//! alignment function. Positions beyond `f(t)` are masked to `-inf`
//! (practically a large negative constant).
//!
//! The causal mask is applied as an additive bias:
//!   `S_masked[t, j] = S[t, j] + mask[t, j]`
//!
//! where `mask[t, j] = 0` for unmasked (j <= f(t)) and `mask[t, j] = MASK_VALUE`
//! for masked positions.
//!
//! Three alignment functions are provided:
//!
//! 1. **Linear**: `f(t) = floor(t * T_enc / T_dec)` — uniform pacing
//! 2. **Strict causal**: `f(t) = min(t, T_enc - 1)` — 1:1 until encoder exhausted
//! 3. **Lookahead**: `f(t) = min(floor(t * T_enc / T_dec) + L, T_enc - 1)` —
//!    linear with L positions of lookahead
//!
//! Part of #1729: Attention Monotonicity Proofs — Phase 21.

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{BoundedTensor, TensorParamBinding};
use ndarray::ArrayD;

use super::common::weights;

// Causal mask + propagation — delegated to super::common (Part of #1970).
pub(super) use super::common::{
    build_causal_mask, build_linear_causal_mask, build_strict_causal_mask, graph_propagate,
    linear_alignment, strict_causal_alignment,
};

// Sinusoidal PE — delegated to super::common (Part of #1970).
pub(super) use super::common::sinusoidal_pe;

// ---------------------------------------------------------------------------
// Alignment functions unique to causal_attention
// ---------------------------------------------------------------------------

/// Linear with lookahead: `f(t) = min(floor(t * T_enc / T_dec) + L, T_enc - 1)`.
///
/// Allows each decoder step to peek `lookahead` positions ahead of the
/// linear alignment. Useful for non-autoregressive decoders that need
/// context from slightly future encoder positions.
pub(super) fn lookahead_alignment(t: usize, t_dec: usize, t_enc: usize, lookahead: usize) -> usize {
    let base = t * t_enc / t_dec;
    (base + lookahead).min(t_enc.saturating_sub(1))
}

// ---------------------------------------------------------------------------
// Causal mask construction — unique variants
// ---------------------------------------------------------------------------

/// Build a causal mask with lookahead alignment.
pub(super) fn build_lookahead_causal_mask(
    t_dec: usize,
    t_enc: usize,
    lookahead: usize,
) -> ArrayD<f32> {
    build_causal_mask(t_dec, t_enc, |t| {
        lookahead_alignment(t, t_dec, t_enc, lookahead)
    })
}

// ---------------------------------------------------------------------------
// Graph builders — scores + causal mask
// ---------------------------------------------------------------------------

/// Build asymmetric cross-attention with additive causal mask (simple variant).
///
/// Q: [t_dec, d] (Variable — decoder hidden states)
/// K: [t_enc, d] (ConstantTensor — encoder text)
/// mask: [t_dec, t_enc] (ConstantTensor — causal mask)
/// Output: S_masked = Q @ K^T / √d + mask → [t_dec, t_enc]
pub(super) fn build_causal_scores_simple(
    t_dec: usize,
    t_enc: usize,
    d: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("causal_scores_simple");
    let q = b.add_input("decoder_hidden", &[t_dec, d]);
    let k = b.add_input("encoder_text", &[t_enc, d]);
    let mask = b.add_input("causal_mask", &[t_dec, t_enc]);

    let scale = 1.0 / (d as f32).sqrt();
    let score_shape = [t_dec, t_enc];
    let scores = b.add_matmul(q, k, true, Some(scale), &score_shape);
    let masked = b.add_binary_add(scores, mask, &score_shape);

    b.build(masked).expect("valid causal score graph")
}

/// Build PE-aware cross-attention with causal mask.
///
/// Q = decoder_hidden + decoder_PE → [t_dec, d]
/// K = encoder_PE → [t_enc, d]
/// mask: [t_dec, t_enc] (ConstantTensor — causal mask)
/// Output: (Q @ K^T / √d) + mask → [t_dec, t_enc]
pub(super) fn build_causal_scores_pe_aware(
    t_dec: usize,
    t_enc: usize,
    d: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("causal_scores_pe");
    let hidden = b.add_input("decoder_hidden", &[t_dec, d]);
    let dec_pe = b.add_input("decoder_pe", &[t_dec, d]);
    let enc_pe = b.add_input("encoder_pe", &[t_enc, d]);
    let mask = b.add_input("causal_mask", &[t_dec, t_enc]);

    let q = b.add_binary_add(hidden, dec_pe, &[t_dec, d]);
    let scale = 1.0 / (d as f32).sqrt();
    let score_shape = [t_dec, t_enc];
    let scores = b.add_matmul(q, enc_pe, true, Some(scale), &score_shape);
    let masked = b.add_binary_add(scores, mask, &score_shape);

    b.build(masked).expect("valid causal PE-aware score graph")
}

// ---------------------------------------------------------------------------
// Binding constructors
// ---------------------------------------------------------------------------

/// Simple causal bindings: Q=Variable, K=ConstantTensor, mask=ConstantTensor.
pub(super) fn simple_causal_bindings(
    t_enc: usize,
    d: usize,
    mask: ArrayD<f32>,
) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(weights::encoder_k(t_enc, d)),
        TensorParamBinding::ConstantTensor(mask),
    ]
}

/// PE-aware causal bindings: Q=Variable, dec_PE=Const, enc_PE=Const, mask=Const.
pub(super) fn pe_aware_causal_bindings(
    t_dec: usize,
    t_enc: usize,
    d: usize,
    pe_scale: f32,
    mask: ArrayD<f32>,
) -> Vec<TensorParamBinding> {
    let mut dec_pe = sinusoidal_pe(t_dec, d);
    let mut enc_pe = sinusoidal_pe(t_enc, d);
    dec_pe.mapv_inplace(|v| v * pe_scale);
    enc_pe.mapv_inplace(|v| v * pe_scale);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(dec_pe),
        TensorParamBinding::ConstantTensor(enc_pe),
        TensorParamBinding::ConstantTensor(mask),
    ]
}

// ---------------------------------------------------------------------------
// Certificate helpers
// ---------------------------------------------------------------------------

/// Extract monotonicity certificate from flat score bounds (diagonal dominance).
///
/// For unmasked or fully-visible attention, the existing certificate checks
/// `S[t,t]` vs all `S[t,j]` for `j != t`.
pub(super) fn extract_certificate(
    output: &BoundedTensor,
    t_dec: usize,
    t_enc: usize,
    input_bound: f64,
    mode: &str,
) -> nn_tts_verify::monotonicity::AttentionMonotonicityCertificate {
    let (lo, hi) = output.lower_upper();
    let score_lower: Vec<f32> = lo.iter().copied().collect();
    let score_upper: Vec<f32> = hi.iter().copied().collect();
    nn_tts_verify::monotonicity::interpret_attention_monotonicity(
        &score_lower,
        &score_upper,
        t_dec,
        t_enc,
        input_bound,
        mode,
    )
    .expect("valid certificate")
}

/// Certificate for causal attention alignment dominance.
///
/// Instead of checking diagonal dominance `S[t,t] > S[t,j]`, checks that
/// the alignment target `S[t, f(t)]` dominates all other unmasked positions:
///   `lower(S[t, f(t)]) > max_{j unmasked, j != f(t)}(upper(S[t, j]))`
///
/// This is the correct monotonicity property for causal TTS attention where
/// the alignment function `f(t)` maps decoder steps to encoder positions.
#[derive(Debug)]
pub(super) struct CausalAlignmentCertificate {
    pub(super) decoder_steps: usize,
    pub(super) encoder_positions: usize,
    pub(super) min_margin: f64,
    pub(super) is_proven: bool,
    pub(super) row_margins: Vec<f64>,
    pub(super) alignment: Vec<usize>,
    pub(super) input_bound: f64,
    pub(super) propagation_mode: String,
}

/// Extract a causal alignment certificate from score bounds.
///
/// For each decoder step `t`, checks that `S[t, f(t)]` dominates all other
/// *unmasked* positions in row `t`. Only rows where `f(t) < t_enc` are checked.
///
/// `alignment_fn(t)` returns the target encoder position for decoder step `t`.
pub(super) fn extract_causal_certificate(
    output: &BoundedTensor,
    t_dec: usize,
    t_enc: usize,
    input_bound: f64,
    mode: &str,
    alignment_fn: impl Fn(usize) -> usize,
) -> CausalAlignmentCertificate {
    let (lo, hi) = output.lower_upper();
    let flat_lo: Vec<f32> = lo.iter().copied().collect();
    let flat_hi: Vec<f32> = hi.iter().copied().collect();

    let mut row_margins = Vec::new();
    let mut alignment = Vec::new();

    for t in 0..t_dec {
        let target = alignment_fn(t);
        if target >= t_enc {
            continue;
        }

        // Skip "post-alignment" rows where the alignment target has
        // saturated at the last encoder position and all positions are
        // visible. These rows represent decoder steps that have consumed
        // all text — alignment dominance isn't meaningful here because
        // there's no single "correct" target.
        let max_visible = target; // alignment_fn(t) = max visible position
        if t_enc > 1 && target == t_enc - 1 && max_visible == t_enc - 1 {
            continue;
        }

        alignment.push(target);

        let target_lo = f64::from(flat_lo[t * t_enc + target]);

        // Find max upper bound among OTHER unmasked positions.
        // A position is "unmasked" if its mask value would be 0 (i.e., j <= target).
        let mut max_other_hi = f64::NEG_INFINITY;
        for j in 0..=max_visible {
            if j != target {
                let upper = f64::from(flat_hi[t * t_enc + j]);
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

    let min_margin = row_margins.iter().copied().fold(f64::INFINITY, f64::min);

    CausalAlignmentCertificate {
        decoder_steps: t_dec,
        encoder_positions: t_enc,
        min_margin,
        is_proven: min_margin > 0.0,
        row_margins,
        alignment,
        input_bound,
        propagation_mode: mode.to_string(),
    }
}

/// Count unmasked positions per row from a causal mask.
///
/// Returns a vector of length `t_dec` where entry `t` is the number of
/// encoder positions that decoder step `t` can attend to (mask[t,j] == 0).
pub(super) fn count_unmasked_per_row(mask: &ArrayD<f32>, t_dec: usize, t_enc: usize) -> Vec<usize> {
    let data = mask.as_slice().expect("contiguous mask");
    (0..t_dec)
        .map(|t| (0..t_enc).filter(|&j| data[t * t_enc + j] == 0.0).count())
        .collect()
}
