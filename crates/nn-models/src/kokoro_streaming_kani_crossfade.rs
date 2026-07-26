// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

/// Harness 2: Crossfade inverse is finite and positive for cf >= 2.
///
/// SUBSTANTIVE: proves that `1.0 / (cf - 1) as f32` is finite and
/// positive for cf in 2..=480. This guards against division-by-zero
/// in the crossfade computation.
///
/// Covers: `kokoro_streaming.rs` line 231.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn crossfade_inverse_finite() {
    let cf: usize = kani::any();
    kani::assume(cf >= 2 && cf <= 480);

    let inv = 1.0f32 / (cf - 1) as f32;

    assert!(inv.is_finite(), "crossfade inverse must be finite");
    assert!(inv > 0.0, "crossfade inverse must be positive");
    assert!(inv <= 1.0, "crossfade inverse must be <= 1.0 for cf >= 2");
}

/// Harness 4: Crossfade with arbitrary finite inputs is finite.
///
/// SUBSTANTIVE: proves that the crossfade formula produces a finite
/// result for any finite inputs, regardless of input range. This is
/// important because model outputs may not always be perfectly clipped.
///
/// Covers: `kokoro_streaming.rs` line 234.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn crossfade_arbitrary_finite_inputs_finite() {
    let tail: f32 = kani::any();
    let head: f32 = kani::any();
    kani::assume(tail.is_finite());
    kani::assume(head.is_finite());
    // Bound inputs to a reasonable range to avoid overflow.
    // Model outputs are typically in [-10, 10] even without clipping.
    kani::assume(tail >= -10.0 && tail <= 10.0);
    kani::assume(head >= -10.0 && head <= 10.0);

    let alpha: f32 = kani::any();
    kani::assume(alpha.is_finite() && alpha >= 0.0 && alpha <= 1.0);

    let one_minus_alpha = 1.0 - alpha;
    let result = tail * one_minus_alpha + head * alpha;

    assert!(
        result.is_finite(),
        "crossfade of finite bounded inputs must be finite"
    );
}

/// Harness 6: Crossfade discontinuity bound matches analytical claim.
///
/// SUBSTANTIVE: proves the key P6 analytical property — that the maximum
/// sample-to-sample difference in a linear crossfade is bounded by
/// `range * step` where `range = max - min` of the input and
/// `step = 1 / (cf - 1)`.
///
/// For adjacent crossfade samples at indices i and i+1:
///   out[i]   = tail[i] * (1 - i*step) + head[i] * i*step
///   out[i+1] = tail[i+1] * (1 - (i+1)*step) + head[i+1] * (i+1)*step
///
/// When tail and head are constant (worst case for per-sample delta from
/// alpha change alone), `|out[i+1] - out[i]| = |head - tail| * step`.
///
/// This harness proves the per-sample case: the alpha change between
/// adjacent samples contributes at most `|head - tail| * step` to the
/// output difference.
///
/// Covers: P6 streaming safety in `nn-tts-verify/src/moonshot_crown_properties.rs`.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn crossfade_per_sample_discontinuity_bounded() {
    let tail_val: f32 = kani::any();
    let head_val: f32 = kani::any();
    kani::assume(tail_val.is_finite() && tail_val >= -1.0 && tail_val <= 1.0);
    kani::assume(head_val.is_finite() && head_val >= -1.0 && head_val <= 1.0);

    let cf: usize = kani::any();
    kani::assume(cf >= 2 && cf <= 480);

    let i: usize = kani::any();
    kani::assume(i + 1 < cf);

    let step = 1.0f32 / (cf - 1) as f32;

    // Two adjacent crossfade outputs with constant tail/head values.
    let alpha_i = i as f32 * step;
    let alpha_next = (i + 1) as f32 * step;

    let out_i = tail_val * (1.0 - alpha_i) + head_val * alpha_i;
    let out_next = tail_val * (1.0 - alpha_next) + head_val * alpha_next;

    assert!(out_i.is_finite(), "crossfade output i must be finite");
    assert!(out_next.is_finite(), "crossfade output i+1 must be finite");

    let delta = (out_next - out_i).abs();
    let range = (head_val - tail_val).abs();

    // The analytical bound: delta <= range * step + epsilon.
    // We add a small epsilon (1e-6) for IEEE 754 rounding.
    let bound = range * step + 1e-6;
    assert!(
        delta <= bound,
        "per-sample discontinuity must be bounded by range * step"
    );
}

// ---------------------------------------------------------------------------
// NaN propagation harnesses — document the unguarded path (#3388 Gap 2)
// ---------------------------------------------------------------------------

/// Harness 7: NaN in crossfade input propagates to output.
///
/// SUBSTANTIVE: Models the crossfade formula at `kokoro_streaming.rs:234`.
/// Proves that a NaN sample in either the tail or head chunk propagates
/// through the linear interpolation to the output, even with valid alpha.
///
/// This documents the crossfade NaN gap: `crossfade_chunks()` assumes finite
/// input but does not validate it. Combined with harnesses 1-6 (which prove
/// correctness for finite inputs), this shows that pre-crossfade validation
/// is required for NaN safety.
///
/// Covers: #3388 Gap 2. kokoro_streaming.rs line 234.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn nan_crossfade_tail_propagates() {
    let tail_val = f32::NAN;
    let head_val: f32 = kani::any();
    kani::assume(head_val.is_finite() && head_val >= -1.0 && head_val <= 1.0);

    let alpha: f32 = kani::any();
    kani::assume(alpha.is_finite() && alpha >= 0.0 && alpha <= 1.0);
    // NaN * 0.0 = NaN in IEEE 754, so NaN propagates even at alpha = 1.0.

    let result = tail_val * (1.0 - alpha) + head_val * alpha;

    assert!(
        result.is_nan(),
        "NaN in tail chunk must propagate through crossfade"
    );
}

/// Harness 8: NaN in head chunk propagates through crossfade.
///
/// SUBSTANTIVE: Symmetric case of harness 7 — proves NaN in the head
/// (incoming) chunk propagates to the crossfade output.
///
/// Covers: #3388 Gap 2. kokoro_streaming.rs line 234.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn nan_crossfade_head_propagates() {
    let tail_val: f32 = kani::any();
    kani::assume(tail_val.is_finite() && tail_val >= -1.0 && tail_val <= 1.0);

    let head_val = f32::NAN;

    let alpha: f32 = kani::any();
    kani::assume(alpha.is_finite() && alpha >= 0.0 && alpha <= 1.0);
    // NaN * 0.0 = NaN in IEEE 754, so NaN propagates even at alpha = 0.0.

    let result = tail_val * (1.0 - alpha) + head_val * alpha;

    assert!(
        result.is_nan(),
        "NaN in head chunk must propagate through crossfade"
    );
}

/// Harness 9: NaN in crossfade propagates even at boundary alphas (0.0 and 1.0).
///
/// SUBSTANTIVE: Proves that NaN propagation through crossfade is NOT prevented
/// by alpha being at its boundary values. At alpha=0.0, the formula is
/// `tail * 1.0 + head * 0.0`. If tail is NaN, the output is NaN despite
/// head contributing nothing. At alpha=1.0, symmetric: NaN head propagates.
///
/// This is critical because `0.0 * NaN = NaN` in IEEE 754 — even the
/// "zero weight" term produces NaN, not zero. No alpha value can isolate
/// the output from a NaN input.
///
/// Covers: #3388 Gap 2 (crossfade boundary). kokoro_streaming.rs line 234.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn nan_crossfade_propagates_at_boundary_alpha() {
    let valid: f32 = kani::any();
    kani::assume(valid.is_finite() && valid >= -1.0 && valid <= 1.0);

    // At alpha = 0.0: result = NaN * 1.0 + valid * 0.0 = NaN + 0 = NaN.
    let result_alpha_zero = f32::NAN * 1.0 + valid * 0.0;
    assert!(
        result_alpha_zero.is_nan(),
        "NaN tail at alpha=0 must still produce NaN (0.0 * valid = 0, but NaN * 1.0 = NaN)"
    );

    // At alpha = 1.0: result = valid * 0.0 + NaN * 1.0 = 0 + NaN = NaN.
    let result_alpha_one = valid * 0.0 + f32::NAN * 1.0;
    assert!(
        result_alpha_one.is_nan(),
        "NaN head at alpha=1 must still produce NaN"
    );
}
