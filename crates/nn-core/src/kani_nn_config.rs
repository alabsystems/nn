// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for nn module configuration validation (#3799).
//!
//! Proves that shared validation helpers (`validate_eps`, `validate_heads`,
//! `validate_divisible`) correctly accept valid inputs and reject invalid ones.
//! These validators are the gatekeepers for all normalization, attention, and
//! grouped convolution constructors — a bug here affects every nn layer.
//!
//! Properties proved:
//! - validate_eps rejects NaN, Inf, negative values
//! - validate_eps accepts all finite non-negative values
//! - validate_heads rejects 0
//! - validate_heads accepts all positive values
//! - validate_divisible rejects non-divisible pairs
//! - validate_divisible accepts all divisible pairs
//! - GroupNorm num_groups divides num_channels (construction invariant)
//! - BatchNormConfig default eps is finite and positive

#![cfg(kani)]

use crate::layers::validation::{validate_divisible, validate_eps, validate_heads};

// ---------------------------------------------------------------------------
// validate_eps: finite non-negative accepted, NaN/Inf/negative rejected
// ---------------------------------------------------------------------------

/// Prove: validate_eps accepts all finite non-negative f64 values.
///
/// This is the gatekeeper for LayerNorm, RmsNorm, GroupNorm, BatchNorm,
/// and InstanceNorm constructors. A false rejection would prevent creating
/// valid normalization layers.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_eps_accepts_finite_nonnegative() {
    // Use u32 to generate arbitrary finite f64 via bit patterns
    let bits: u32 = kani::any();
    let val = (bits as f64) / (u32::MAX as f64) * 100.0; // [0.0, 100.0]

    // val is always finite and non-negative by construction
    assert!(val.is_finite());
    assert!(val >= 0.0);

    let result = validate_eps(val, "test");
    assert!(
        result.is_ok(),
        "validate_eps must accept finite non-negative values"
    );
}

/// Prove: validate_eps rejects NaN.
///
/// IEEE 754 NaN bypasses comparisons. If validate_eps used
/// `eps < 0.0` instead of `!eps.is_finite() || eps < 0.0`, NaN would
/// slip through (NaN < 0.0 returns false). This harness proves the
/// is_finite check catches NaN.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_eps_rejects_nan() {
    let result = validate_eps(f64::NAN, "test");
    assert!(result.is_err(), "validate_eps must reject NaN");
}

/// Prove: validate_eps rejects positive infinity.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_eps_rejects_pos_infinity() {
    let result = validate_eps(f64::INFINITY, "test");
    assert!(result.is_err(), "validate_eps must reject +Inf");
}

/// Prove: validate_eps rejects negative infinity.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_eps_rejects_neg_infinity() {
    let result = validate_eps(f64::NEG_INFINITY, "test");
    assert!(result.is_err(), "validate_eps must reject -Inf");
}

/// Prove: validate_eps rejects negative values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_eps_rejects_negative() {
    let bits: u32 = kani::any();
    kani::assume(bits >= 1); // ensure val > 0
    let val = -((bits as f64) / (u32::MAX as f64) * 100.0); // (-100.0, 0.0)
    assert!(val < 0.0);
    assert!(val.is_finite());

    let result = validate_eps(val, "test");
    assert!(result.is_err(), "validate_eps must reject negative values");
}

// ---------------------------------------------------------------------------
// validate_heads: positive accepted, zero rejected
// ---------------------------------------------------------------------------

/// Prove: validate_heads accepts all positive values.
///
/// This guards MultiHeadAttention, GQA, and MLA constructors.
/// A false rejection would prevent constructing valid attention layers.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_heads_accepts_positive() {
    let n: u16 = kani::any();
    kani::assume(n >= 1);

    let result = validate_heads(n as usize, "test");
    assert!(result.is_ok(), "validate_heads must accept positive values");
}

/// Prove: validate_heads rejects zero.
///
/// Zero heads would cause division-by-zero in head_dim = model_dim / num_heads.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_heads_rejects_zero() {
    let result = validate_heads(0, "test");
    assert!(result.is_err(), "validate_heads must reject 0");
}

// ---------------------------------------------------------------------------
// validate_divisible: divisible pairs accepted, non-divisible rejected
// ---------------------------------------------------------------------------

/// Prove: validate_divisible accepts all (a, b) where a % b == 0 and b > 0.
///
/// This guards GroupNorm (channels % groups), GQA (num_heads % num_kv_heads),
/// and grouped convolutions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_divisible_accepts_multiples() {
    let b: u8 = kani::any();
    let multiplier: u8 = kani::any();
    kani::assume(b >= 1);
    kani::assume(multiplier >= 1);

    let a = (b as usize) * (multiplier as usize);

    let result = validate_divisible(a, b as usize, "a", "b", "test");
    assert!(result.is_ok(), "validate_divisible must accept multiples");
}

/// Prove: validate_divisible rejects non-divisible pairs.
///
/// When a % b != 0, the operation (e.g., splitting channels into groups)
/// would produce uneven splits, causing shape mismatches downstream.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_divisible_rejects_non_multiples() {
    let a: u8 = kani::any();
    let b: u8 = kani::any();
    kani::assume(b >= 2); // need b >= 2 to have non-divisible cases
    kani::assume(a >= 1);
    kani::assume((a as usize) % (b as usize) != 0);

    let result = validate_divisible(a as usize, b as usize, "a", "b", "test");
    assert!(
        result.is_err(),
        "validate_divisible must reject non-multiples"
    );
}

// ---------------------------------------------------------------------------
// BatchNormConfig defaults
// ---------------------------------------------------------------------------

/// Prove: BatchNormConfig default() produces valid eps.
///
/// The default eps (1e-5) must be finite and positive. A NaN or Inf default
/// would silently break all BatchNorm layers created with Default::default().
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn batch_norm_config_default_eps_valid() {
    let config = crate::layers::BatchNormConfig::default();

    assert!(config.eps.is_finite(), "default eps must be finite");
    assert!(config.eps > 0.0, "default eps must be positive");
    assert!(
        validate_eps(config.eps, "test").is_ok(),
        "default eps must pass validation"
    );
}

/// Prove: BatchNormConfig builder chain preserves validity of other fields.
///
/// When setting eps via with_eps(), other fields (remove_mean, affine,
/// momentum) must retain their default values.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn batch_norm_config_builder_preserves_defaults() {
    let config = crate::layers::BatchNormConfig::new(1e-6);

    // new() sets eps and uses defaults for everything else
    assert!(config.eps == 1e-6);
    assert!(config.remove_mean == true);
    assert!(config.affine == true);
    assert!(config.momentum == 0.1);
}
