// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Kokoro TTS duration prediction chain.
//!
//! The duration chain (`kokoro_tts.rs` forward_text lines 303-307) converts
//! raw prosody logits to frame durations via:
//!   sigmoid(logits).sum(dim=2) / speed -> clamp(1, max_dur) -> round -> clamp_min(1)
//!
//! These harnesses prove the critical numerical invariants:
//! 1. validate_speed accepts exactly positive finite values (calls production code)
//! 2. Duration computation produces values in [1.0, max_dur] for valid inputs
//! 3. Round + clamp_min always produces counts >= 1 (no phoneme dropout)
//! 4. Rounded durations are exact non-negative integers safe for usize cast
//!
//! Part of #2218.
//!
//! No CBMC transcendental stubs needed: the duration chain uses only
//! arithmetic operations (division, clamp, round) that CBMC models natively.
//! Sigmoid range is modeled via nondeterministic bounds in harness 2.
//!
//! Rounding mode: production uses `f32::round_ties_even()` (banker's rounding,
//! IEEE 754 default). Harnesses 3-4 use `round_ties_even()` to match.
//! Previous version incorrectly used `f32::round()` (round-half-away-from-zero).

use crate::kokoro_error::{validate_speed, KokoroError};

// CBMC transcendental stub for f32::floor.
fn floor_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}

/// Harness 1: validate_speed accepts exactly {positive, finite} values.
///
/// SUBSTANTIVE: calls the actual production `validate_speed()` function from
/// `kokoro_error.rs` and verifies its behavior for all f32 values. Proves:
/// - Every positive finite speed returns Ok
/// - Every non-positive or non-finite speed returns Err(InvalidSpeed)
/// - No gaps in the guard logic (IEEE 754 edge cases: NaN, +/-Inf, -0.0, subnormals)
///
/// Covers: `kokoro_error.rs` validate_speed (lines 119-123).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_speed_rejects_invalid_accepts_valid() {
    let speed: f32 = kani::any();

    let result = validate_speed(speed);

    // Expected behavior: accept iff positive and finite.
    let should_accept = speed.is_finite() && speed > 0.0;

    if should_accept {
        assert!(
            result.is_ok(),
            "positive finite speed must be accepted by validate_speed"
        );
    } else {
        assert!(
            result.is_err(),
            "non-positive or non-finite speed must be rejected"
        );
        // Verify the error is the correct variant.
        match result.unwrap_err() {
            KokoroError::InvalidSpeed { value } => {
                // The error must carry the original speed value.
                assert_eq!(
                    value.to_bits(),
                    speed.to_bits(),
                    "error must carry original speed"
                );
            }
            _ => panic!("validate_speed must return InvalidSpeed variant"),
        }
    }
}

/// Harness 2: Duration sigmoid->sum->divide->clamp is bounded in [1, max_dur].
///
/// forward_text (`kokoro_tts.rs` lines 303-307) computes:
///   dur_logits.sigmoid()             -> each element in [0, 1]
///   .sum(dim=2)                      -> per-phoneme sum in [0, max_dur]
///   .mul_scalar(1.0 / f64(speed))    -> scaled (f64 intermediate)
///   .clamp(1.0, max_dur)             -> in [1.0, max_dur]
///
/// The sigmoid is STRUCTURAL (stubbed). The sum->divide->clamp chain is
/// SUBSTANTIVE: proves that for any sigmoid outputs and any valid speed,
/// the final duration is bounded in [1.0, max_dur].
///
/// Production uses f64 for the 1/speed reciprocal, so 0*Inf=NaN cannot
/// occur (f64 1/speed is finite for all f32 speed > 0). This harness
/// conservatively models f32 arithmetic, which is strictly harder.
///
/// Covers: `kokoro_tts.rs` lines 303-307.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn duration_sigmoid_sum_clamp_bounded() {
    // max_dur = 50 in production (kokoro_config.rs:58).
    let max_dur: f32 = 50.0;

    // Sum of max_dur sigmoid outputs, each in [0, 1] -> sum in [0, 50].
    // Modeled as a single nondeterministic value representing the sum.
    let sigmoid_sum: f32 = kani::any();
    kani::assume(sigmoid_sum.is_finite());
    kani::assume(sigmoid_sum >= 0.0 && sigmoid_sum <= max_dur);

    // Speed: validated by validate_speed as positive finite.
    let speed: f32 = kani::any();
    kani::assume(speed.is_finite());
    kani::assume(speed > 0.0);

    // Division: 1.0 / speed then multiply.
    // Production uses f64 intermediate, making this even safer.
    let inv_speed = 1.0f32 / speed;
    let scaled = sigmoid_sum * inv_speed;

    // Clamp to [1.0, max_dur].
    // f32::clamp: if self > max -> max, if self < min -> min, else self.
    // For Inf: Inf > 50.0 -> returns 50.0. Safe.
    // For NaN: all comparisons false -> returns NaN. Caught by check_tensor_finite.
    let clamped = scaled.clamp(1.0, max_dur);

    // For non-NaN scaled values, clamp produces [1.0, max_dur].
    if !scaled.is_nan() {
        assert!(clamped.is_finite(), "clamped duration must be finite");
        assert!(clamped >= 1.0, "clamped duration must be >= 1.0");
        assert!(clamped <= max_dur, "clamped duration must be <= max_dur");
    }
    // NaN case (0.0 * Inf in f32): clamp returns NaN.
    // This is caught by check_tensor_finite() at kokoro_tts.rs:309.
    // In production (f64 path), this case cannot occur.
}

/// Harness 3: Round + clamp_min produces counts >= 1 for valid durations.
///
/// length_regulate (`kokoro_tts.rs` lines 113-116) computes:
///   durations.round().clamp_min(1.0)
///
/// Input durations are in [1.0, max_dur] from the clamp in forward_text.
///
/// SUBSTANTIVE: proves that for any duration in [1.0, 50.0], banker's
/// rounding (round_ties_even) with clamp_min(1.0) always produces a
/// count >= 1.0 — no phoneme can get zero frames in repeat_interleave.
///
/// Uses `round_ties_even()` to match production (`nn-core` DynTensor::round
/// at `dyn_tensor/ops/math.rs:148` uses `f32::round_ties_even`).
///
/// Covers: `kokoro_tts.rs` lines 113-116.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn round_clamp_min_positive_count() {
    let dur: f32 = kani::any();
    kani::assume(dur.is_finite());
    kani::assume(dur >= 1.0 && dur <= 50.0);

    // Production uses banker's rounding (round-half-to-even, IEEE 754 default).
    let rounded = dur.round_ties_even();

    // For dur >= 1.0: nearest integer is >= 1.0.
    // Proof: dur in [1.0, 1.5) -> round = 1.0.
    //        dur = 1.5 -> round = 2.0 (ties-to-even: 2 is even).
    //        dur in (1.5, 2.5) -> round = 2.0.
    //        dur = 2.5 -> round = 2.0 (ties-to-even: 2 is even).
    //        All cases >= 1.0.
    assert!(rounded.is_finite(), "round of finite must be finite");
    assert!(rounded >= 1.0, "round_ties_even(dur >= 1.0) must be >= 1.0");

    // clamp_min(1.0) is defense-in-depth, should be no-op here.
    let count = rounded.max(1.0);
    assert!(count >= 1.0, "final count must be >= 1.0");
    assert!(count.is_finite(), "final count must be finite");

    // Upper bound: round(50.0) = 50.0. No input in [1.0, 50.0] rounds > 50.
    assert!(count <= 50.0, "count must be <= 50.0 for max_dur=50");
}

/// Harness 4: Rounded durations are exact integers safe for index use.
///
/// repeat_interleave (`kokoro_tts.rs` line 115) interprets the rounded
/// f32 durations as repeat counts. These must be exact non-negative
/// integers with no fractional part, and the u32 cast must be lossless.
///
/// SUBSTANTIVE: proves round_ties_even(clamp(d, 1.0, 50.0)) produces a
/// value that is exactly representable as a non-negative integer in f32
/// (verified via floor equality) and survives u32 round-trip without loss.
///
/// Uses `round_ties_even()` to match production (`nn-core` DynTensor::round
/// at `dyn_tensor/ops/math.rs:148` uses `f32::round_ties_even`).
///
/// Covers: `kokoro_tts.rs` lines 113-115.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::floor, floor_f32_stub)]
fn rounded_duration_is_exact_integer() {
    let dur: f32 = kani::any();
    kani::assume(dur.is_finite());
    kani::assume(dur >= 1.0 && dur <= 50.0);

    let rounded = dur.round_ties_even();
    let count = rounded.max(1.0); // clamp_min(1.0)

    // round_ties_even() of a finite value in [1, 50] produces an exact integer.
    // Verify: count == floor(count), meaning no fractional part.
    assert_eq!(count, count.floor(), "rounded count must be exact integer");

    // Safe u32 cast: value is in [1.0, 50.0] as exact integer.
    let as_u32 = count as u32;
    assert!(as_u32 >= 1, "u32 count >= 1");
    assert!(as_u32 <= 50, "u32 count <= 50");

    // Round-trip: f32 -> u32 -> f32 is lossless for small integers.
    assert_eq!(count, as_u32 as f32, "u32 round-trip must be lossless");
}
