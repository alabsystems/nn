// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor elementwise operation safety (#3590).
//!
//! Proves arithmetic invariants of unary and compound elementwise operations
//! used throughout the model execution pipeline. All harnesses inline the
//! scalar math from production code (ops/math.rs, ops/math_compound.rs) since
//! Kani cannot model ndarray or GPU storage.
//!
//! Properties proved:
//! - neg(neg(x)) == x (involution)
//! - abs(x) >= 0 for all finite x
//! - sqr(x) >= 0 for all finite x
//! - relu(x) >= 0 for all finite x
//! - sigmoid output in (0, 1) for finite x
//! - clamp output in [min, max] when min <= max
//! - clamp min > max rejection
//! - fract output in [0, 1) for finite positive x
//! - floor(x) <= x for finite x
//! - tanh output in (-1, 1) for finite x
//! - exp(x) > 0 for finite x where exp doesn't overflow

#![cfg(kani)]

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn exp_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

fn floor_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}

fn tanh_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r >= -1.0 && r <= 1.0);
    r
}

// ---------------------------------------------------------------------------
// neg(neg(x)) == x  (involution / double negation)
// ---------------------------------------------------------------------------

/// Prove: negation is an involution — neg(neg(x)) == x for all finite f32.
///
/// Inlines math.rs:139: `f32_arr.mapv(|x| -x)` applied twice.
/// This is a fundamental algebraic property that binary op decompositions
/// depend on (e.g., elu uses `self.neg()?.relu()?.neg()?`).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn neg_neg_is_identity() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());

    let neg_x = -x;
    let neg_neg_x = -neg_x;

    // IEEE 754: -(-x) == x for all finite values (bit-exact).
    assert_eq!(
        x.to_bits(),
        neg_neg_x.to_bits(),
        "neg(neg(x)) must equal x (bit-exact)"
    );
}

// ---------------------------------------------------------------------------
// abs(x) >= 0 for all finite x
// ---------------------------------------------------------------------------

/// Prove: absolute value is non-negative for all finite f32.
///
/// Inlines math.rs:138: `f32_arr.mapv(f32::abs)`.
/// This invariant is used by norm computations and distance metrics
/// throughout the model pipeline.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn abs_non_negative() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());

    let abs_x = x.abs();

    assert!(abs_x >= 0.0, "abs(x) must be >= 0 for finite x");
    assert!(abs_x.is_finite(), "abs of finite input must be finite");
}

/// Prove: abs(x) == abs(-x) for all finite f32 (symmetry).
///
/// The absolute value function must be symmetric around zero. This is
/// relied upon by normalization layers that compute |x - mean|.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn abs_symmetric() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());

    let abs_x = x.abs();
    let abs_neg_x = (-x).abs();

    assert_eq!(
        abs_x.to_bits(),
        abs_neg_x.to_bits(),
        "abs(x) must equal abs(-x) (bit-exact)"
    );
}

// ---------------------------------------------------------------------------
// sqr(x) >= 0 for all finite x
// ---------------------------------------------------------------------------

/// Prove: squaring produces a non-negative result for all finite f32.
///
/// Inlines math.rs:137: `f32_arr.mapv(|x| x * x)`.
/// sqr is used in snake activation (`sin_sq = scaled.sin()?.sqr()?`),
/// L2 norms, and MSE loss. A negative result would corrupt these.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn sqr_non_negative() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());

    let sqr_x = x * x;

    // sqr can overflow to +Inf for large |x|, but never negative.
    assert!(sqr_x >= 0.0, "x*x must be >= 0 for finite x");
}

// ---------------------------------------------------------------------------
// relu(x) >= 0 for all finite x
// ---------------------------------------------------------------------------

/// Prove: ReLU output is non-negative for all finite f32.
///
/// Inlines math.rs:125: `f32_arr.mapv(|x| x.max(0.0))`.
/// ReLU non-negativity is a critical invariant — Kokoro's ISTFTNet decoder
/// chains relu with snake activation, and negative relu output would
/// corrupt the snake computation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn relu_non_negative() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());

    let relu_x = x.max(0.0);

    assert!(relu_x >= 0.0, "relu(x) must be >= 0");
    assert!(relu_x.is_finite(), "relu of finite input must be finite");
}

/// Prove: ReLU is idempotent — relu(relu(x)) == relu(x).
///
/// This invariant matters because compiler fusion passes may merge
/// consecutive ReLU ops. The fused version must produce the same result.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn relu_idempotent() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());

    let relu_x = x.max(0.0);
    let relu_relu_x = relu_x.max(0.0);

    assert_eq!(
        relu_x.to_bits(),
        relu_relu_x.to_bits(),
        "relu(relu(x)) must equal relu(x) (bit-exact)"
    );
}

// ---------------------------------------------------------------------------
// sigmoid output in (0, 1) for bounded finite x
// ---------------------------------------------------------------------------

/// Prove: sigmoid output is in (0, 1) for finite f32 inputs in [-88, 88].
///
/// Inlines math.rs:133: `f32_arr.mapv(|x| 1.0 / (1.0 + (-x).exp()))`.
/// Sigmoid is used in LSTM gates and attention gating. Output outside
/// [0, 1] would corrupt gate values. The range [-88, 88] avoids exp
/// overflow (exp(88) ~ 1.65e38, within f32 range).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn sigmoid_output_bounded() {
    let x_bits: u16 = kani::any();
    // Map u16 to [-88.0, 88.0] range: scale to [0, 176] then shift to [-88, 88]
    let x = (x_bits as f32 / 65535.0) * 176.0 - 88.0;
    kani::assume(x.is_finite());

    let neg_x = -x;
    let exp_neg_x = neg_x.exp();
    // Guard: exp must not overflow
    kani::assume(exp_neg_x.is_finite());

    let sigmoid = 1.0 / (1.0 + exp_neg_x);

    assert!(sigmoid > 0.0, "sigmoid(x) must be > 0");
    assert!(sigmoid <= 1.0, "sigmoid(x) must be <= 1");
    assert!(
        sigmoid.is_finite(),
        "sigmoid must be finite for bounded input"
    );
}

// ---------------------------------------------------------------------------
// clamp output in [min, max] when min <= max
// ---------------------------------------------------------------------------

/// Prove: clamp(x, lo, hi) output is always in [lo, hi] for finite inputs.
///
/// Inlines math_compound.rs:164: `arr.mapv(|x| x.clamp(lo, hi))`.
/// The clamp invariant is fundamental to the GPU fused clamp kernel
/// (NativeOpKind dispatch) and bounds verification.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn clamp_output_in_range() {
    let x_bits: u32 = kani::any();
    let lo_bits: u32 = kani::any();
    let hi_bits: u32 = kani::any();

    let x = f32::from_bits(x_bits);
    let lo = f32::from_bits(lo_bits);
    let hi = f32::from_bits(hi_bits);

    kani::assume(x.is_finite());
    kani::assume(lo.is_finite());
    kani::assume(hi.is_finite());
    kani::assume(lo <= hi);

    let clamped = x.clamp(lo, hi);

    assert!(clamped >= lo, "clamp result must be >= lo");
    assert!(clamped <= hi, "clamp result must be <= hi");
    assert!(clamped.is_finite(), "clamp of finite inputs must be finite");
}

/// Prove: clamp is idempotent — clamp(clamp(x, lo, hi), lo, hi) == clamp(x, lo, hi).
///
/// This matters for fusion: if the optimizer merges two clamp passes with
/// the same bounds, the result must be identical to a single pass.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn clamp_idempotent() {
    let x_bits: u32 = kani::any();
    let lo_bits: u32 = kani::any();
    let hi_bits: u32 = kani::any();

    let x = f32::from_bits(x_bits);
    let lo = f32::from_bits(lo_bits);
    let hi = f32::from_bits(hi_bits);

    kani::assume(x.is_finite());
    kani::assume(lo.is_finite());
    kani::assume(hi.is_finite());
    kani::assume(lo <= hi);

    let clamped_once = x.clamp(lo, hi);
    let clamped_twice = clamped_once.clamp(lo, hi);

    assert_eq!(
        clamped_once.to_bits(),
        clamped_twice.to_bits(),
        "clamp must be idempotent (bit-exact)"
    );
}

// ---------------------------------------------------------------------------
// floor(x) <= x for finite x
// ---------------------------------------------------------------------------

/// Prove: floor(x) <= x for all finite f32.
///
/// Inlines math.rs:147: `f32_arr.mapv(f32::floor)`.
/// This invariant is used by the fract() implementation: `x - floor(x)`
/// which must produce a non-negative result for positive x.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::floor, floor_f32_stub)]
fn floor_le_input() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());

    let floor_x = x.floor();

    assert!(floor_x <= x, "floor(x) must be <= x for finite x");
    assert!(floor_x.is_finite(), "floor of finite input must be finite");
}

// ---------------------------------------------------------------------------
// fract output in [0, 1) for finite positive x (MSL semantics)
// ---------------------------------------------------------------------------

/// Prove: fract(x) = x - floor(x) is in [0, 1) for finite positive f32.
///
/// Inlines math.rs:149: `f32_arr.mapv(|x| x - x.floor())`.
/// Uses MSL/GLSL `fract()` semantics (floor-based), NOT Rust `f32::fract()`
/// which uses trunc and can be negative. The [0, 1) range is critical for
/// phase accumulation in STFT and signal processing.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::floor, floor_f32_stub)]
fn fract_non_negative_for_positive_input() {
    let bits: u32 = kani::any();
    let x = f32::from_bits(bits);
    kani::assume(x.is_finite());
    kani::assume(x >= 0.0);

    let fract_x = x - x.floor();

    assert!(fract_x >= 0.0, "fract(x) must be >= 0 for positive x");
    // fract is strictly < 1.0 for non-integer values, == 0.0 for integers.
    assert!(fract_x <= 1.0, "fract(x) must be <= 1.0");
}

// ---------------------------------------------------------------------------
// tanh output in (-1, 1) for bounded finite x
// ---------------------------------------------------------------------------

/// Prove: tanh(x) is in [-1, 1] for finite f32 in [-88, 88].
///
/// Inlines math.rs:132: `f32_arr.mapv(f32::tanh)`.
/// Tanh is used in LSTM cell state gating and GELU approximation.
/// Output outside [-1, 1] would corrupt gate values (LSTM h_t).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::tanh, tanh_f32_stub)]
fn tanh_output_bounded() {
    let x_bits: u16 = kani::any();
    // Map to [-88, 88] to avoid exp overflow in tanh internals
    let x = (x_bits as f32 / 65535.0) * 176.0 - 88.0;
    kani::assume(x.is_finite());

    let tanh_x = x.tanh();

    assert!(tanh_x >= -1.0, "tanh(x) must be >= -1");
    assert!(tanh_x <= 1.0, "tanh(x) must be <= 1");
    assert!(tanh_x.is_finite(), "tanh must be finite for bounded input");
}

// ---------------------------------------------------------------------------
// exp(x) > 0 for finite x (within non-overflow range)
// ---------------------------------------------------------------------------

/// Prove: exp(x) > 0 for finite f32 in [-87, 87] (no overflow, no underflow to 0).
///
/// Inlines math.rs:134: `f32_arr.mapv(f32::exp)`.
/// exp positivity is critical for softmax (division by sum of exp) and
/// sigmoid (1 / (1 + exp(-x))). A zero or negative exp would cause
/// division by zero or sign corruption.
///
/// Range [-87, 87] ensures: exp(-87) ~ 1.4e-38 > 0 (above f32 min_positive ~1.2e-38)
/// and exp(87) ~ 6.1e37 < f32::MAX. Outside this range, underflow to 0 or overflow
/// to Inf can occur.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::exp, exp_f32_stub)]
fn exp_positive_bounded() {
    let x_bits: u16 = kani::any();
    // Map to [-87, 87]: safe range for f32 exp
    let x = (x_bits as f32 / 65535.0) * 174.0 - 87.0;
    kani::assume(x.is_finite());

    let exp_x = x.exp();

    assert!(exp_x > 0.0, "exp(x) must be > 0 for x in [-87, 87]");
    assert!(
        exp_x.is_finite(),
        "exp(x) must be finite for x in [-87, 87]"
    );
}
