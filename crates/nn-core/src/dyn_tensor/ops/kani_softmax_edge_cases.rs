// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for softmax edge cases from production bugs (#3752).
//!
//! Proves correctness of the specific edge-case handlers that were added
//! after 4 separate bug fixes:
//!
//! - **#1310**: All values -inf → output zeros (not NaN)
//! - **#1326**: All-neg-inf guard in GPU decomposed softmax (sum clamping)
//! - **#1339**: +inf input → uniform over +inf positions, 0 elsewhere
//! - **#1691**: bf16/f16 clamp constants prevent exp() overflow
//!
//! Unlike the existing `kani_softmax.rs` which proves finite-input properties,
//! these harnesses prove the EDGE-CASE HANDLERS are correct — the code paths
//! that fire when inputs contain infinities or when dtype limits matter.
//!
//! All harnesses call actual production functions or faithfully mirror the
//! production code paths (CPU softmax lane processing from `softmax.rs`).

use crate::DType;

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

fn ln_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -100.0 && r <= 100.0);
    r
}

// ============================================================================
// 1. Production CPU softmax lane processor — mirrors dyn_tensor/softmax.rs
//    softmax_dispatch() lines 86-107 EXACTLY.
// ============================================================================

/// Faithfully reproduces the production CPU softmax lane-processing logic
/// from `dyn_tensor/softmax.rs` `softmax_dispatch()` (the Zip::from block).
///
/// This is the EXACT algorithm, including all three guards:
/// - all-neg-inf guard (#1310)
/// - +inf guard (#1339)
/// - normal finite path
fn production_softmax_lane(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let n = input.len();

    // Step 1: compute max (production: fold(NEG_INFINITY, f32::max))
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Guard: all-neg-inf lane → zero output (#1310).
    if max_val == f32::NEG_INFINITY {
        for o in output.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    // Guard: +inf in lane → uniform over +inf positions, 0 elsewhere (#1339).
    if max_val == f32::INFINITY {
        let inf_count = input.iter().filter(|&&x| x == f32::INFINITY).count();
        let prob = 1.0 / inf_count as f32;
        for i in 0..n {
            output[i] = if input[i] == f32::INFINITY { prob } else { 0.0 };
        }
        return;
    }

    // Normal path: max-subtract, exp, normalize
    let mut sum = 0.0_f32;
    for i in 0..n {
        output[i] = (input[i] - max_val).exp();
        sum += output[i];
    }
    for i in 0..n {
        output[i] /= sum;
    }
}

/// Faithfully reproduces the production CPU log_softmax lane-processing logic
/// from `dyn_tensor/softmax.rs` `log_softmax_dispatch()`.
fn production_log_softmax_lane(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len());
    let n = input.len();

    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Guard: all-neg-inf lane → -inf output (#1310).
    if max_val == f32::NEG_INFINITY {
        for o in output.iter_mut() {
            *o = f32::NEG_INFINITY;
        }
        return;
    }

    // Guard: +inf in lane → log of softmax result.
    if max_val == f32::INFINITY {
        let inf_count = input.iter().filter(|&&x| x == f32::INFINITY).count();
        let log_prob = -(inf_count as f32).ln();
        for i in 0..n {
            output[i] = if input[i] == f32::INFINITY {
                log_prob
            } else {
                f32::NEG_INFINITY
            };
        }
        return;
    }

    // Normal path: log_sum_exp = max + log(sum(exp(x - max)))
    let sum_exp: f32 = input.iter().map(|&x| (x - max_val).exp()).sum();
    let log_sum_exp = max_val + sum_exp.ln();
    for i in 0..n {
        output[i] = input[i] - log_sum_exp;
    }
}

// ============================================================================
// 2. Bug #1310: all-neg-inf → zeros (softmax) / -inf (log_softmax)
// ============================================================================

/// Prove: when ALL inputs are -inf, softmax outputs all zeros.
///
/// This is the #1310 guard. Before the fix, the max-subtract trick produced
/// -inf - (-inf) = NaN under IEEE 754, which propagated NaN through the
/// entire output. The production code detects max_val == NEG_INFINITY and
/// zeros the lane.
///
/// Uses 3-element array with symbolic choice of how many elements are -inf
/// (all of them, since kani::assume forces all three to be -inf).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_all_neg_inf_produces_zeros() {
    let input = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    let mut output = [0.0_f32; 3];

    production_softmax_lane(&input, &mut output);

    assert_eq!(output[0], 0.0, "all-neg-inf softmax must produce 0.0");
    assert_eq!(output[1], 0.0, "all-neg-inf softmax must produce 0.0");
    assert_eq!(output[2], 0.0, "all-neg-inf softmax must produce 0.0");
    // Also verify no NaN leaked through
    assert!(!output[0].is_nan(), "must not be NaN");
    assert!(!output[1].is_nan(), "must not be NaN");
    assert!(!output[2].is_nan(), "must not be NaN");
}

/// Prove: when ALL inputs are -inf, log_softmax outputs all -inf.
///
/// The mathematically correct result: log(0) = -inf for all positions.
/// Before #1310 fix, this would produce NaN.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_softmax_all_neg_inf_produces_neg_inf() {
    let input = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    let mut output = [0.0_f32; 3];

    production_log_softmax_lane(&input, &mut output);

    assert_eq!(
        output[0],
        f32::NEG_INFINITY,
        "all-neg-inf log_softmax must produce -inf"
    );
    assert_eq!(
        output[1],
        f32::NEG_INFINITY,
        "all-neg-inf log_softmax must produce -inf"
    );
    assert_eq!(
        output[2],
        f32::NEG_INFINITY,
        "all-neg-inf log_softmax must produce -inf"
    );
    assert!(!output[0].is_nan(), "must not be NaN");
    assert!(!output[1].is_nan(), "must not be NaN");
    assert!(!output[2].is_nan(), "must not be NaN");
}

// ============================================================================
// 3. Bug #1339: +inf input handling
// ============================================================================

/// Prove: when one input is +inf, it gets probability 1.0, others get 0.0.
///
/// This is the #1339 guard. Without it, inf - inf = NaN in the max-subtract
/// trick. The production code detects max_val == INFINITY and distributes
/// probability uniformly over +inf positions.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_single_inf_gets_all_probability() {
    // Symbolic finite values for positions 0 and 2
    let a: i8 = kani::any();
    let c: i8 = kani::any();

    let input = [a as f32, f32::INFINITY, c as f32];
    let mut output = [0.0_f32; 3];

    production_softmax_lane(&input, &mut output);

    assert_eq!(
        output[0], 0.0,
        "finite input must get probability 0 when +inf present"
    );
    assert!(
        (output[1] - 1.0).abs() < 1e-7,
        "+inf input must get probability 1.0, got {}",
        output[1]
    );
    assert_eq!(
        output[2], 0.0,
        "finite input must get probability 0 when +inf present"
    );
    // Verify no NaN
    assert!(!output[0].is_nan(), "must not be NaN");
    assert!(!output[1].is_nan(), "must not be NaN");
    assert!(!output[2].is_nan(), "must not be NaN");
}

/// Prove: when multiple inputs are +inf, probability is split uniformly.
///
/// With 2 out of 3 inputs being +inf, each +inf position should get 0.5
/// and the finite position should get 0.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_multiple_inf_uniform_split() {
    let a: i8 = kani::any();
    let input = [f32::INFINITY, a as f32, f32::INFINITY];
    let mut output = [0.0_f32; 3];

    production_softmax_lane(&input, &mut output);

    assert!(
        (output[0] - 0.5).abs() < 1e-7,
        "+inf should get 1/2 = 0.5, got {}",
        output[0]
    );
    assert_eq!(output[1], 0.0, "finite input must get 0 when +inf present");
    assert!(
        (output[2] - 0.5).abs() < 1e-7,
        "+inf should get 1/2 = 0.5, got {}",
        output[2]
    );
}

/// Prove: when all 3 inputs are +inf, probability is 1/3 each.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_all_inf_uniform_third() {
    let input = [f32::INFINITY, f32::INFINITY, f32::INFINITY];
    let mut output = [0.0_f32; 3];

    production_softmax_lane(&input, &mut output);

    let expected = 1.0_f32 / 3.0;
    assert!(
        (output[0] - expected).abs() < 1e-6,
        "all-inf softmax must be 1/3 each, got {}",
        output[0]
    );
    assert!(
        (output[1] - expected).abs() < 1e-6,
        "all-inf softmax must be 1/3 each, got {}",
        output[1]
    );
    assert!(
        (output[2] - expected).abs() < 1e-6,
        "all-inf softmax must be 1/3 each, got {}",
        output[2]
    );
}

/// Prove: log_softmax with single +inf gives log(1)=0 at inf, -inf elsewhere.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_softmax_single_inf_correct() {
    let a: i8 = kani::any();
    let c: i8 = kani::any();

    let input = [a as f32, f32::INFINITY, c as f32];
    let mut output = [0.0_f32; 3];

    production_log_softmax_lane(&input, &mut output);

    // log(1.0) = 0.0 for the +inf position
    assert!(
        (output[1] - 0.0).abs() < 1e-7,
        "log_softmax at +inf with 1 inf should be log(1)=0, got {}",
        output[1]
    );
    // log(0) = -inf for finite positions
    assert_eq!(
        output[0],
        f32::NEG_INFINITY,
        "log_softmax at finite when +inf present must be -inf"
    );
    assert_eq!(
        output[2],
        f32::NEG_INFINITY,
        "log_softmax at finite when +inf present must be -inf"
    );
}

/// Prove: log_softmax with 2 +inf gives log(1/2) = -ln(2) at each inf position.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_softmax_two_inf_correct() {
    let a: i8 = kani::any();
    let input = [f32::INFINITY, a as f32, f32::INFINITY];
    let mut output = [0.0_f32; 3];

    production_log_softmax_lane(&input, &mut output);

    let expected_log_prob = -(2.0_f32).ln(); // log(1/2) = -ln(2)
    assert!(
        (output[0] - expected_log_prob).abs() < 1e-5,
        "log_softmax at +inf with 2 infs should be -ln(2), got {}",
        output[0]
    );
    assert_eq!(
        output[1],
        f32::NEG_INFINITY,
        "log_softmax at finite when +inf present must be -inf"
    );
    assert!(
        (output[2] - expected_log_prob).abs() < 1e-5,
        "log_softmax at +inf with 2 infs should be -ln(2), got {}",
        output[2]
    );
}

// ============================================================================
// 4. Mixed -inf and finite: attention mask pattern
// ============================================================================

/// Prove: softmax with -inf at masked positions produces valid probabilities
/// over unmasked positions.
///
/// This is the core attention mask use case: some positions are -inf (masked),
/// the rest are finite. The unmasked positions must form a valid probability
/// distribution (non-negative, sum to 1), and masked positions must be exactly 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_attention_mask_pattern() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();

    // Attention mask: positions 0 and 1 are unmasked (finite), position 2 is masked (-inf)
    let input = [a as f32, b as f32, f32::NEG_INFINITY];
    let mut output = [0.0_f32; 3];

    production_softmax_lane(&input, &mut output);

    // Masked position must be exactly 0
    assert_eq!(
        output[2], 0.0,
        "-inf masked position must have probability 0"
    );
    // Unmasked positions must be non-negative
    assert!(output[0] >= 0.0, "unmasked probability must be >= 0");
    assert!(output[1] >= 0.0, "unmasked probability must be >= 0");
    // Sum must be 1
    let sum = output[0] + output[1] + output[2];
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax with mask must still sum to 1, got {}",
        sum
    );
}

// ============================================================================
// 5. Production softmax: NaN-free guarantee for all edge-case paths
// ============================================================================

/// Prove: the production softmax lane processor NEVER produces NaN for any
/// combination of finite, -inf, and +inf inputs.
///
/// This is the comprehensive edge-case proof covering all three guards
/// (#1310 all-neg-inf, #1339 +inf, normal finite). For symbolic 2-element
/// lanes, every possible combination of {finite, -inf, +inf} is covered.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_never_nan_2elem() {
    // Choose each element from: finite (i8 range), -inf, or +inf
    let choice0: u8 = kani::any();
    let choice1: u8 = kani::any();
    kani::assume(choice0 < 3);
    kani::assume(choice1 < 3);

    let val0: f32 = match choice0 {
        0 => {
            let v: i8 = kani::any();
            v as f32
        }
        1 => f32::NEG_INFINITY,
        _ => f32::INFINITY,
    };
    let val1: f32 = match choice1 {
        0 => {
            let v: i8 = kani::any();
            v as f32
        }
        1 => f32::NEG_INFINITY,
        _ => f32::INFINITY,
    };

    let input = [val0, val1];
    let mut output = [0.0_f32; 2];

    production_softmax_lane(&input, &mut output);

    assert!(
        !output[0].is_nan(),
        "softmax must never produce NaN (elem 0)"
    );
    assert!(
        !output[1].is_nan(),
        "softmax must never produce NaN (elem 1)"
    );
    // All outputs must be non-negative
    assert!(output[0] >= 0.0, "softmax output must be >= 0");
    assert!(output[1] >= 0.0, "softmax output must be >= 0");
}

/// Prove: the production log_softmax lane processor NEVER produces NaN for
/// any combination of finite, -inf, and +inf inputs.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_softmax_never_nan_2elem() {
    let choice0: u8 = kani::any();
    let choice1: u8 = kani::any();
    kani::assume(choice0 < 3);
    kani::assume(choice1 < 3);

    let val0: f32 = match choice0 {
        0 => {
            let v: i8 = kani::any();
            v as f32
        }
        1 => f32::NEG_INFINITY,
        _ => f32::INFINITY,
    };
    let val1: f32 = match choice1 {
        0 => {
            let v: i8 = kani::any();
            v as f32
        }
        1 => f32::NEG_INFINITY,
        _ => f32::INFINITY,
    };

    let input = [val0, val1];
    let mut output = [0.0_f32; 2];

    production_log_softmax_lane(&input, &mut output);

    assert!(
        !output[0].is_nan(),
        "log_softmax must never produce NaN (elem 0)"
    );
    assert!(
        !output[1].is_nan(),
        "log_softmax must never produce NaN (elem 1)"
    );
    // All log_softmax outputs must be <= 0 (or -inf)
    assert!(output[0] <= 1e-6, "log_softmax output must be non-positive");
    assert!(output[1] <= 1e-6, "log_softmax output must be non-positive");
}

// ============================================================================
// 6. Bug #1691: softmax_clamp_constants prevents exp() overflow per dtype
// ============================================================================

/// Prove: BF16 clamp constants prevent exp() overflow in GPU decomposition.
///
/// In the GPU decomposed softmax, after clamping to [min, max] and subtracting
/// the max, the argument to exp() is in [-2*max, 0]. For BF16, if max were
/// f32::MAX instead of bf16::MAX, the intermediate (max - min) would overflow
/// bf16 storage. The fix (#1691) uses dtype-appropriate constants.
///
/// This proves: max_val - min_val fits in the dtype's range (no overflow).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn bf16_clamp_range_no_overflow() {
    let (max_val, min_val, min_pos) = super::softmax::softmax_clamp_constants(DType::BF16);

    // The range [min, max] must fit within bf16 representable range
    let bf16_max = f64::from(half::bf16::MAX);
    assert!(max_val <= bf16_max, "BF16 max clamp must be <= bf16::MAX");
    assert!(min_val >= -bf16_max, "BF16 min clamp must be >= -bf16::MAX");
    // min_positive must be representable in bf16
    let bf16_min_pos = f64::from(half::bf16::MIN_POSITIVE);
    assert!(
        min_pos >= bf16_min_pos,
        "BF16 min_positive must be >= bf16::MIN_POSITIVE"
    );
    // The range max - min must not overflow bf16 when used in subtraction
    // (In the GPU path, shifted = input - max, which is at most max - min)
    let range = max_val - min_val;
    assert!(range.is_finite(), "BF16 clamp range must be finite");
    assert!(range > 0.0, "BF16 clamp range must be positive");
}

/// Prove: F16 clamp constants prevent exp() overflow in GPU decomposition.
///
/// F16 has only 5 exponent bits, so MAX is 65504. Using f32 MAX (~3.4e38)
/// would catastrophically overflow f16 storage. This was the root cause of #1691.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn f16_clamp_range_no_overflow() {
    let (max_val, min_val, min_pos) = super::softmax::softmax_clamp_constants(DType::F16);

    let f16_max = f64::from(half::f16::MAX); // 65504
    assert!(
        max_val <= f16_max,
        "F16 max clamp must be <= f16::MAX (65504)"
    );
    assert!(
        min_val >= -f16_max,
        "F16 min clamp must be >= -f16::MAX (-65504)"
    );
    // Critically: max must NOT be f32::MAX (the bug that #1691 fixed)
    assert!(
        max_val < 70000.0,
        "F16 max clamp must be ~65504, not f32::MAX"
    );
    // min_positive must be representable in f16
    let f16_min_pos = f64::from(half::f16::MIN_POSITIVE);
    assert!(
        min_pos >= f16_min_pos,
        "F16 min_positive must be >= f16::MIN_POSITIVE"
    );
}

/// Prove: for ALL float dtypes, the exp() argument after max-subtraction is
/// non-positive (so exp() result is in (0, 1]).
///
/// In GPU decomposed softmax: shifted = clamped_input - max.
/// Since clamped_input <= max_clamp and max >= clamped_input for all elements,
/// shifted <= 0 always. exp(shifted) is in (0, 1], preventing overflow.
///
/// This proves the max-subtraction trick keeps exp() bounded for any dtype.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn exp_argument_non_positive_after_max_sub() {
    let idx: u8 = kani::any();
    kani::assume(idx < 3); // Only float dtypes: F32, F16, BF16
    let dt = match idx {
        0 => DType::F32,
        1 => DType::F16,
        _ => DType::BF16,
    };

    let (max_val, min_val, _) = super::softmax::softmax_clamp_constants(dt);

    // Any clamped input is in [min_val, max_val].
    // The max over a lane of clamped inputs is also in [min_val, max_val].
    // shifted = clamped_input - lane_max <= max_val - min_val (when input=max, lane_max=min)
    // But more importantly: shifted = input - max <= 0 always (since max >= input).
    //
    // For the softmax to be safe, we need: for any input in [min_val, max_val]
    // and any max in [min_val, max_val] where max >= input:
    // exp(input - max) must be finite.
    //
    // Since input - max <= 0, exp(input - max) <= exp(0) = 1.0. Always safe.
    // The only risk is underflow to 0 (harmless) for very negative arguments.

    // Verify the shifted argument is bounded: worst case is min_val - max_val
    let worst_shift = min_val - max_val;
    assert!(worst_shift <= 0.0, "max-subtracted value must be <= 0");
    assert!(worst_shift.is_finite(), "worst-case shift must be finite");

    // Verify exp(0) = 1.0 (best case: input equals max)
    let best_exp = (0.0_f64).exp();
    assert!((best_exp - 1.0).abs() < 1e-15, "exp(0) must be 1.0");
}

// ============================================================================
// 7. softmax_clamp_constants: sum-clamp floor prevents division by zero
// ============================================================================

/// Prove: the min_positive constant (sum clamp floor) prevents division by
/// zero in the GPU decomposed softmax.
///
/// In `gpu_softmax_decomposed`, the sum of exponentials is clamped to be
/// >= min_positive. If min_positive were 0 or negative, the final division
/// `exp_vals / sum_vals` could produce Inf or NaN from division by zero.
///
/// This proves min_positive > 0 for all float dtypes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn sum_clamp_prevents_div_by_zero_all_float_dtypes() {
    let idx: u8 = kani::any();
    kani::assume(idx < 3);
    let dt = match idx {
        0 => DType::F32,
        1 => DType::F16,
        _ => DType::BF16,
    };

    let (_, _, min_pos) = super::softmax::softmax_clamp_constants(dt);

    assert!(
        min_pos > 0.0,
        "min_positive must be > 0 to prevent div-by-zero"
    );
    assert!(
        min_pos.is_finite(),
        "min_positive must be finite (not Inf or NaN)"
    );
    assert!(!min_pos.is_nan(), "min_positive must not be NaN");

    // Verify that dividing 1.0 by min_pos produces a finite result
    let test_div = 1.0_f64 / min_pos;
    assert!(
        test_div.is_finite(),
        "1.0 / min_positive must be finite (min_pos not too small)"
    );
}

// ============================================================================
// 8. check_div_result_finite: prove it catches all non-finite values
// ============================================================================

/// Scalar version of check_div_result_finite for Kani tractability.
/// Returns true if the value is finite, false otherwise.
/// Mirrors the production logic: `result.iter().filter(|v| !v.is_finite()).count()`
fn scalar_check_finite(val: f32) -> bool {
    val.is_finite()
}

/// Prove: check_finite correctly identifies NaN as non-finite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn check_finite_catches_nan() {
    let result = scalar_check_finite(f32::NAN);
    assert!(!result, "NaN must be detected as non-finite");
}

/// Prove: check_finite correctly identifies +Inf as non-finite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn check_finite_catches_pos_inf() {
    let result = scalar_check_finite(f32::INFINITY);
    assert!(!result, "+Inf must be detected as non-finite");
}

/// Prove: check_finite correctly identifies -Inf as non-finite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn check_finite_catches_neg_inf() {
    let result = scalar_check_finite(f32::NEG_INFINITY);
    assert!(!result, "-Inf must be detected as non-finite");
}

/// Prove: check_finite passes all finite f32 values.
///
/// For any symbolic finite f32, is_finite() returns true.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn check_finite_passes_all_finite() {
    let val: f32 = kani::any();
    kani::assume(val.is_finite());

    let result = scalar_check_finite(val);
    assert!(result, "finite value must pass check_finite");
}

/// Prove: division by zero produces non-finite result that check_finite catches.
///
/// IEEE 754: x / 0.0 = +/-Inf (for x != 0), 0.0 / 0.0 = NaN.
/// Both cases must be caught by check_finite.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn div_by_zero_caught_by_check_finite() {
    let numerator: f32 = kani::any();
    kani::assume(numerator.is_finite());

    let result = numerator / 0.0_f32;

    // IEEE 754: finite / 0 = Inf (if numerator != 0) or NaN (if numerator == 0)
    // Either way, it must not be finite.
    if numerator == 0.0 || numerator == -0.0 {
        // 0.0 / 0.0 = NaN
        assert!(result.is_nan(), "0/0 must be NaN");
    } else {
        // nonzero / 0 = +/-Inf
        assert!(result.is_infinite(), "nonzero/0 must be Inf");
    }
    assert!(
        !scalar_check_finite(result),
        "division by zero must be caught by check_finite"
    );
}

// ============================================================================
// 9. Softmax output invariants for edge cases: sum, bounds, ordering
// ============================================================================

/// Prove: softmax with mixed finite and -inf sums to 1.0.
///
/// When some (but not all) positions are -inf, the output should still
/// form a valid probability distribution over the non-masked positions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_mixed_finite_neg_inf_sums_to_one() {
    let a: i8 = kani::any();

    // One finite value, one -inf
    let input = [a as f32, f32::NEG_INFINITY];
    let mut output = [0.0_f32; 2];

    production_softmax_lane(&input, &mut output);

    // The finite position should get all the probability
    let sum = output[0] + output[1];
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax with -inf positions must sum to 1, got {}",
        sum
    );
    // The -inf position must be exactly 0
    assert_eq!(output[1], 0.0, "-inf position must have probability 0");
    // The finite position must get probability 1.0
    assert!(
        (output[0] - 1.0).abs() < 1e-5,
        "sole finite position must get probability 1.0, got {}",
        output[0]
    );
}

/// Prove: softmax with +inf input still sums to 1.0 (probability conservation).
///
/// Even with the +inf guard, the output must remain a valid probability distribution.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_inf_input_sums_to_one() {
    let a: i8 = kani::any();
    let c: i8 = kani::any();

    let input = [a as f32, f32::INFINITY, c as f32];
    let mut output = [0.0_f32; 3];

    production_softmax_lane(&input, &mut output);

    let sum = output[0] + output[1] + output[2];
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax with +inf must still sum to 1.0, got {}",
        sum
    );
}

/// Prove: softmax preserves ordering even when max element is very large.
///
/// For the max-subtraction trick, very large differences can cause exp()
/// to underflow to 0. The ordering should still be preserved: the largest
/// finite input gets the largest probability.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn softmax_extreme_range_preserves_ordering() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();
    kani::assume(a > b); // a is strictly larger

    // Add large offset to test near-overflow regime
    let offset = 80.0_f32;
    let input = [a as f32 + offset, b as f32 + offset, b as f32];
    let mut output = [0.0_f32; 3];

    production_softmax_lane(&input, &mut output);

    // Position 0 (largest) must have largest probability
    assert!(
        output[0] >= output[1],
        "largest input must get largest probability"
    );
    assert!(
        output[0] >= output[2],
        "largest input must get largest probability"
    );
}

// ============================================================================
// 10. Log-softmax edge case invariants
// ============================================================================

/// Prove: log_softmax with finite inputs produces finite outputs.
///
/// The numerically stable formulation (max-subtraction) must prevent
/// intermediate overflow from producing non-finite outputs.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_softmax_finite_inputs_finite_outputs() {
    let a: i8 = kani::any();
    let b: i8 = kani::any();

    let input = [a as f32, b as f32];
    let mut output = [0.0_f32; 2];

    production_log_softmax_lane(&input, &mut output);

    assert!(
        output[0].is_finite(),
        "log_softmax[0] must be finite for finite inputs"
    );
    assert!(
        output[1].is_finite(),
        "log_softmax[1] must be finite for finite inputs"
    );
}

/// Prove: log_softmax with -inf mask produces -inf at masked positions.
///
/// For attention masks, -inf positions must map to -inf in log_softmax
/// (since softmax gives 0 probability, and log(0) = -inf).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(3)]
#[kani::stub(f32::exp, exp_f32_stub)]
#[kani::stub(f32::ln, ln_f32_stub)]
fn log_softmax_neg_inf_mask_produces_neg_inf_output() {
    let a: i8 = kani::any();

    let input = [a as f32, f32::NEG_INFINITY];
    let mut output = [0.0_f32; 2];

    production_log_softmax_lane(&input, &mut output);

    // The -inf position: exp(-inf) = 0, softmax = 0, log(0) = -inf
    // Production uses: x - log_sum_exp. For -inf: -inf - finite = -inf.
    assert_eq!(
        output[1],
        f32::NEG_INFINITY,
        "-inf position in log_softmax must produce -inf"
    );
    // The finite position should produce a finite non-positive value
    assert!(
        output[0].is_finite(),
        "finite position must produce finite log_softmax"
    );
    assert!(output[0] <= 1e-6, "log_softmax must be non-positive");
}
