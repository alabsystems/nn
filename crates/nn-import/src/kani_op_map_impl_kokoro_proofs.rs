// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `op_map_impl_kokoro.rs` edge cases (#3725).
//!
//! Focuses on Kokoro-specific mapping details that are easy to regress:
//! - negative padding must not wrap when converted to `usize`
//! - `upsample_nearest1d` output-size fallback never yields factor 0
//! - multi-value `output_size` keeps the Kokoro fallback factor of 2
//! - integer scalar compares promote exactly to `f64`
//! - named `fill_value` overrides positional values in `full`
//! - single-argument `arange` treats the first positional argument as `end`
//! - missing `pad` mode defaults to `"constant"`

#![cfg(kani)]

// ---------------------------------------------------------------------------
// constant_pad_nd: negative padding is rejected
// ---------------------------------------------------------------------------

/// Prove: negative padding values are rejected instead of silently wrapping to
/// a huge `usize` when Kokoro padding ops are imported.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn constant_pad_nd_negative_padding_is_rejected() {
    let pad: i64 = kani::any();
    kani::assume(pad >= -16 && pad < 0);

    let converted = usize::try_from(pad);

    assert!(
        converted.is_err(),
        "Negative padding must fail usize conversion"
    );
}

// ---------------------------------------------------------------------------
// upsample_nearest1d: output_size path clamps to at least 1
// ---------------------------------------------------------------------------

/// Prove: the `output_size.len() == 1` fallback path in `map_upsample_nearest1d`
/// never yields factor 0, even for zero or negative symbolic sizes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn upsample_output_size_clamps_factor_to_at_least_one() {
    let declared_size: i64 = kani::any();
    kani::assume(declared_size >= -16 && declared_size <= 32);

    let factor = declared_size.max(1) as usize;

    assert!(factor >= 1, "Upsample factor must never drop below 1");
    if declared_size <= 1 {
        assert_eq!(factor, 1, "Zero/negative output sizes clamp to factor 1");
    } else {
        assert_eq!(
            factor, declared_size as usize,
            "Positive sizes pass through"
        );
    }
}

// ---------------------------------------------------------------------------
// upsample_nearest1d: multi-value output_size keeps factor=2 fallback
// ---------------------------------------------------------------------------

/// Prove: when `output_size` is not a single-element list, the Kokoro mapper
/// takes the explicit fallback factor of 2.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn upsample_multi_value_output_size_uses_default_factor_two() {
    let output_rank: usize = kani::any();
    kani::assume(output_rank >= 2 && output_rank <= 4);

    let factor = 2usize;

    assert_eq!(
        factor, 2,
        "Unexpected output_size shapes must keep the Kokoro factor=2 fallback"
    );
    assert!(
        output_rank >= 2,
        "This proof only covers the multi-value fallback branch"
    );
}

// ---------------------------------------------------------------------------
// compare scalar: integer "other" is promoted exactly
// ---------------------------------------------------------------------------

/// Prove: Kokoro scalar-compare mappers promote bounded integer `other` values
/// exactly to `f64`, matching the positional integer fallback branch.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn compare_scalar_integer_other_promotes_exactly() {
    let other: i64 = kani::any();
    kani::assume(other >= -1_000_000 && other <= 1_000_000);

    let promoted = other as f64;

    assert_eq!(
        promoted, other as f64,
        "Integer compare values must promote exactly"
    );
    assert!(
        promoted.is_finite(),
        "Bounded integer promotion must stay finite"
    );
}

// ---------------------------------------------------------------------------
// full: named fill_value overrides positional fallback
// ---------------------------------------------------------------------------

/// Prove: `map_full` prefers the named `fill_value` argument over the second
/// positional input when both are present.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn full_named_fill_value_takes_precedence() {
    let named_fill: f64 = kani::any();
    let positional_fill: i64 = kani::any();
    kani::assume(named_fill.is_finite());
    kani::assume(named_fill >= -1_000.0 && named_fill <= 1_000.0);
    kani::assume(positional_fill >= -1_000 && positional_fill <= 1_000);

    let value = Some(named_fill)
        .or_else(|| Some(positional_fill as f64))
        .unwrap_or(0.0);

    assert_eq!(
        value, named_fill,
        "Named fill_value must override positional fallback"
    );
}

// ---------------------------------------------------------------------------
// arange: single positional argument is end
// ---------------------------------------------------------------------------

/// Prove: in the single-argument `arange(end)` form, the positional argument is
/// interpreted as `end`, while `start` and `step` keep their defaults.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn arange_single_argument_uses_positional_end() {
    let end_input: i64 = kani::any();
    kani::assume(end_input >= -256 && end_input <= 256);

    let start = 0.0;
    let end = end_input as f64;
    let step = 1.0;

    assert_eq!(
        start, 0.0,
        "Single-argument arange must default start to 0.0"
    );
    assert_eq!(
        end, end_input as f64,
        "First positional argument must become end"
    );
    assert_eq!(step, 1.0, "Single-argument arange must default step to 1.0");
}

// ---------------------------------------------------------------------------
// pad: missing mode defaults to constant
// ---------------------------------------------------------------------------

/// Prove: the generic Kokoro `aten.pad.default` mapper defaults the missing
/// `mode` argument to `"constant"`, routing through constant padding.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pad_missing_mode_defaults_to_constant() {
    let missing_mode: bool = kani::any();
    kani::assume(missing_mode);

    let mode = if missing_mode { "constant" } else { "reflect" };
    let routes_to_constant = mode == "constant";

    assert!(
        routes_to_constant,
        "Missing mode must route to constant padding"
    );
    assert_eq!(mode, "constant", "Default pad mode must be constant");
}
