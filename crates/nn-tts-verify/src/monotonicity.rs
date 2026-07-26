// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Monotonicity verification for TTS models.
//!
//! Two complementary properties prevent word skipping/repetition in TTS:
//!
//! ## 1. Duration Positivity (non-autoregressive TTS, e.g. Kokoro)
//!
//! Proves that duration predictor outputs are bounded below, guaranteeing
//! every phoneme receives non-zero synthesis time:
//!
//! ```text
//! ∀ text_features ∈ [-B, B]^(d_model × T), style ∈ [-B, B]^style_dim:
//!     dur_logits[i] > threshold   (e.g., -10.0)
//! ```
//!
//! Since `duration = exp(dur_logits)`, a lower bound on `dur_logits > -C`
//! guarantees `duration > exp(-C) > 0`.
//!
//! ## 2. Attention Score Diagonal Dominance (autoregressive TTS)
//!
//! For cross-attention score matrices `S[t, j]` (pre-softmax), proves that
//! diagonal elements dominate off-diagonal elements:
//!
//! ```text
//! ∀ inputs ∈ [-B, B]: lower(S[t, t]) > upper(S[t, j]) for all j ≠ t
//! ```
//!
//! Diagonal dominance is a sufficient condition for monotonic attention:
//! if pre-softmax scores have the diagonal as the largest element in each
//! row, softmax concentrates probability mass on the diagonal, ensuring
//! the attention peak at decoder step `t` aligns with encoder position `t`.
//!
//! This prevents word skipping (`argmax(attn[t])` jumps forward) and
//! word repetition (`argmax(attn[t])` stays at `argmax(attn[t-1])`).
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_tts_verify::monotonicity::{
//!     DurationPositivityCertificate, interpret_duration_positivity,
//!     AttentionMonotonicityCertificate, interpret_attention_monotonicity,
//! };
//!
//! // Duration positivity (Kokoro ProsodyPredictor):
//! let cert = interpret_duration_positivity(
//!     crown_lower_bound, -10.0, 1.0, 1.0, 1, "CROWN",
//! );
//! assert!(cert.is_proven);
//!
//! // Attention monotonicity (cross-attention scores):
//! let cert = interpret_attention_monotonicity(
//!     &score_lower_bounds, // [T_dec × T_enc] lower bounds on pre-softmax scores
//!     &score_upper_bounds, // [T_dec × T_enc] upper bounds
//!     4,                   // decoder_steps (T_dec)
//!     4,                   // encoder_positions (T_enc)
//!     1.0,                 // input bound
//!     "CROWN",
//! ).unwrap();
//! assert!(cert.is_proven);
//! ```

fn normalized_propagation_mode(mode: &str) -> String {
    mode.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Classify a propagation provenance string as the sound CROWN family.
///
/// This normalizes common presentation variants such as `alpha-CROWN`,
/// `AlphaCrown`, and `beta_crown` for soundness checks while leaving the
/// original provenance string untouched on certificates.
#[must_use]
pub fn propagation_mode_is_sound_crown_family(mode: &str) -> bool {
    matches!(
        normalized_propagation_mode(mode).as_str(),
        "crown" | "alphacrown" | "betacrown"
    )
}

/// Certificate from a duration positivity proof.
///
/// Records the CROWN lower bound on `dur_logits` and whether it exceeds
/// the minimum threshold. When `is_proven` is true, the model is formally
/// verified to never produce zero-duration phonemes within the input bounds.
#[derive(Debug, Clone)]
pub struct DurationPositivityCertificate {
    /// CROWN lower bound on dur_logits output (scalar, minimum over all elements).
    pub lower_bound: f64,
    /// Minimum acceptable dur_logits value (e.g., -10.0).
    ///
    /// Chosen so that `exp(threshold)` is still positive and non-negligible.
    /// Common values: -10.0 (exp(-10) ≈ 4.5e-5), -20.0 (exp(-20) ≈ 2.1e-9).
    pub threshold: f64,
    /// Text feature input bound (symmetric: `[-bound, +bound]`).
    pub input_bound: f64,
    /// Style embedding input bound (symmetric: `[-bound, +bound]`).
    pub style_bound: f64,
    /// Number of timesteps verified (T).
    ///
    /// T=1 is the baseline (single-step LSTM, zero-init state).
    /// T=4 covers the "transient" region and is the ICLR submission target.
    pub sequence_length: usize,
    /// Whether the proof succeeded: `lower_bound > threshold`.
    pub is_proven: bool,
    /// Propagation method used (`"IBP"`, `"CROWN"`, or `"alpha-CROWN"`).
    pub propagation_mode: String,
}

impl DurationPositivityCertificate {
    /// Whether this certificate came from a sound CROWN-family propagation.
    #[must_use]
    pub fn is_sound_crown_family(&self) -> bool {
        propagation_mode_is_sound_crown_family(&self.propagation_mode)
    }
}

/// Interpret a CROWN lower bound on duration logits as a positivity certificate.
///
/// # Arguments
///
/// * `lower_bound` — minimum value of `dur_logits` from CROWN propagation
/// * `threshold` — minimum acceptable value (e.g., -10.0, since `exp(-10) ≈ 4.5e-5 > 0`)
/// * `input_bound` — symmetric bound on text features (`[-B, B]`)
/// * `style_bound` — symmetric bound on style embedding (`[-B, B]`)
/// * `sequence_length` — number of timesteps (T) in the verification
/// * `propagation_mode` — `"IBP"`, `"CROWN"`, etc.
pub fn interpret_duration_positivity(
    lower_bound: f64,
    threshold: f64,
    input_bound: f64,
    style_bound: f64,
    sequence_length: usize,
    propagation_mode: &str,
) -> DurationPositivityCertificate {
    DurationPositivityCertificate {
        lower_bound,
        threshold,
        input_bound,
        style_bound,
        sequence_length,
        is_proven: lower_bound > threshold,
        propagation_mode: propagation_mode.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Attention score diagonal dominance
// ---------------------------------------------------------------------------

/// Certificate from an attention monotonicity proof via diagonal dominance.
///
/// For a cross-attention score matrix `S[t, j]` (pre-softmax), records whether
/// diagonal elements dominate off-diagonal elements in every row. When
/// `is_proven` is true, `argmax(softmax(S[t, :]))` equals `t` for all
/// decoder steps, formally guaranteeing monotonic attention alignment.
#[derive(Debug, Clone)]
pub struct AttentionMonotonicityCertificate {
    /// Number of decoder steps (rows in the attention matrix).
    pub decoder_steps: usize,
    /// Number of encoder positions (columns in the attention matrix).
    pub encoder_positions: usize,
    /// Minimum margin: `min_t(lower(S[t,t]) - max_{j≠t}(upper(S[t,j])))`.
    ///
    /// Positive = proven monotonic for all rows.
    pub min_margin: f64,
    /// Whether the proof succeeded: `min_margin > 0`.
    pub is_proven: bool,
    /// Per-row margins: `lower(S[t,t]) - max_{j≠t}(upper(S[t,j]))` for each `t`.
    /// Length = `min(decoder_steps, encoder_positions)` (only diagonal-existing rows).
    pub row_margins: Vec<f64>,
    /// Input bound for the Q (decoder) Variable (symmetric `[-B, B]`).
    pub input_bound: f64,
    /// Propagation method used (`"IBP"`, `"CROWN"`, etc.).
    pub propagation_mode: String,
}

impl AttentionMonotonicityCertificate {
    /// Whether this certificate came from a sound CROWN-family propagation.
    #[must_use]
    pub fn is_sound_crown_family(&self) -> bool {
        propagation_mode_is_sound_crown_family(&self.propagation_mode)
    }
}

/// Interpret CROWN bounds on pre-softmax attention scores as a monotonicity
/// certificate via diagonal dominance.
///
/// For each decoder step `t`, checks whether `lower(S[t, t]) > max_{j≠t}(upper(S[t, j]))`.
/// If this holds for all `t`, the attention pattern is formally monotonic:
/// softmax concentrates mass on position `t` for decoder step `t`.
///
/// # Arguments
///
/// * `score_lower` — lower bounds on `S[t, j]`, flattened row-major `[T_dec × T_enc]`
/// * `score_upper` — upper bounds on `S[t, j]`, flattened row-major `[T_dec × T_enc]`
/// * `decoder_steps` — number of decoder steps (rows)
/// * `encoder_positions` — number of encoder positions (columns)
/// * `input_bound` — symmetric bound on Q input
/// * `propagation_mode` — `"IBP"`, `"CROWN"`, etc.
///
/// # Errors
///
/// Returns [`TtsVerifyError::DimensionMismatch`] if slice lengths don't match
/// `decoder_steps * encoder_positions`.
pub fn interpret_attention_monotonicity(
    score_lower: &[f32],
    score_upper: &[f32],
    decoder_steps: usize,
    encoder_positions: usize,
    input_bound: f64,
    propagation_mode: &str,
) -> Result<AttentionMonotonicityCertificate, crate::error::TtsVerifyError> {
    let n = decoder_steps * encoder_positions;
    if score_lower.len() != n {
        return Err(crate::error::TtsVerifyError::DimensionMismatch {
            expected: n,
            actual: score_lower.len(),
            context: "score_lower",
        });
    }
    if score_upper.len() != n {
        return Err(crate::error::TtsVerifyError::DimensionMismatch {
            expected: n,
            actual: score_upper.len(),
            context: "score_upper",
        });
    }

    // Defense-in-depth: reject NaN/Inf in input bounds before computing margins.
    // IEEE 754 NaN comparison (e.g. `upper > max_offdiag_hi`) returns false for NaN,
    // silently skipping corrupted elements and potentially producing false `is_proven`.
    if score_lower.iter().any(|x| !f64::from(*x).is_finite()) {
        return Err(crate::error::TtsVerifyError::InvalidConfig(
            crate::error::InvalidConfigKind::NonFinite {
                param: "score_lower",
            },
        ));
    }
    if score_upper.iter().any(|x| !f64::from(*x).is_finite()) {
        return Err(crate::error::TtsVerifyError::InvalidConfig(
            crate::error::InvalidConfigKind::NonFinite {
                param: "score_upper",
            },
        ));
    }

    let diag_count = decoder_steps.min(encoder_positions);
    let mut row_margins = Vec::with_capacity(diag_count);

    for t in 0..diag_count {
        let diag_lo = f64::from(score_lower[t * encoder_positions + t]);

        // Find maximum upper bound among off-diagonal elements in row t
        let mut max_offdiag_hi = f64::NEG_INFINITY;
        for j in 0..encoder_positions {
            if j != t {
                let upper = f64::from(score_upper[t * encoder_positions + j]);
                if upper > max_offdiag_hi {
                    max_offdiag_hi = upper;
                }
            }
        }

        // If only one column, off-diagonal is empty: trivially monotonic.
        let margin = if encoder_positions <= 1 {
            f64::INFINITY
        } else {
            diag_lo - max_offdiag_hi
        };
        row_margins.push(margin);
    }

    // Use NaN-propagating min: if any row margin is NaN (shouldn't happen after
    // input guards, but defense-in-depth), the overall min_margin will be NaN
    // and `is_proven` will be false (NaN > 0.0 == false).
    let min_margin =
        crate::stats::fold_min_propagate_nan(row_margins.iter().copied(), f64::INFINITY);

    Ok(AttentionMonotonicityCertificate {
        decoder_steps,
        encoder_positions,
        min_margin,
        is_proven: min_margin > 0.0,
        row_margins,
        input_bound,
        propagation_mode: propagation_mode.to_string(),
    })
}

/// Construct an [`AttentionMonotonicityCertificate`] from multi-head
/// weight-space margins (post-softmax).
///
/// Bridges Phase 23's softmax-inclusive attention weight proofs into the
/// moonshot P3 upgrade path. For each decoder step, the per-row margin is
/// the *minimum* across all heads — monotonicity requires dominance in
/// *every* head simultaneously.
///
/// # Arguments
///
/// * `per_head_row_margins` — `[H][T_dec]` margins in weight space per head.
///   `per_head_row_margins[h][t]` = `lower(W[h,t,f(t)]) - max_{j≠f(t)} upper(W[h,t,j])`.
/// * `decoder_steps` — number of decoder steps (T_dec)
/// * `encoder_positions` — number of encoder positions (T_enc)
/// * `input_bound` — symmetric bound on Q input
/// * `propagation_mode` — `"IBP"`, `"CROWN"`, etc.
pub fn from_multi_head_weight_margins(
    per_head_row_margins: &[Vec<f64>],
    decoder_steps: usize,
    encoder_positions: usize,
    input_bound: f64,
    propagation_mode: &str,
) -> Result<AttentionMonotonicityCertificate, crate::error::TtsVerifyError> {
    // Defense-in-depth: reject NaN/Inf in per-head margins.
    // IEEE 754 `m < min_across_heads` returns false when m is NaN,
    // silently skipping the corrupted head and producing a false positive.
    for head_margins in per_head_row_margins {
        if head_margins.iter().any(|m| !m.is_finite()) {
            return Err(crate::error::TtsVerifyError::InvalidConfig(
                crate::error::InvalidConfigKind::NonFinite {
                    param: "per_head_row_margins",
                },
            ));
        }
    }

    // For each decoder step, take the minimum margin across all heads.
    let diag_count = decoder_steps.min(encoder_positions);

    // Fail-closed: every head must provide margins for all diagonal steps.
    // Previously, short margin vectors were silently skipped, leaving the
    // default INFINITY in place — a fail-open soundness gap (#1994).
    for head_margins in per_head_row_margins {
        if head_margins.len() < diag_count {
            return Err(crate::error::TtsVerifyError::DimensionMismatch {
                expected: diag_count,
                actual: head_margins.len(),
                context: "per_head_row_margins",
            });
        }
    }

    let mut row_margins = Vec::with_capacity(diag_count);

    for t in 0..diag_count {
        let mut min_across_heads = f64::INFINITY;
        for head_margins in per_head_row_margins {
            let m = head_margins[t];
            if m < min_across_heads {
                min_across_heads = m;
            }
        }
        row_margins.push(min_across_heads);
    }

    let min_margin =
        crate::stats::fold_min_propagate_nan(row_margins.iter().copied(), f64::INFINITY);

    Ok(AttentionMonotonicityCertificate {
        decoder_steps,
        encoder_positions,
        min_margin,
        is_proven: min_margin > 0.0,
        row_margins,
        input_bound,
        propagation_mode: propagation_mode.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Weight magnitude validation (Phase 44-46 bridge)
// ---------------------------------------------------------------------------

#[path = "monotonicity_weight.rs"]
mod weight;
pub use weight::{
    max_provable_input_bound, validate_weight_magnitudes, WeightMagnitudeCertificate,
};

#[cfg(test)]
#[path = "monotonicity_tests.rs"]
mod tests;
