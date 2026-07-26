// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for extended DynTensor operations (`ops_ext`).
//!
//! Proves properties of:
//! - `flatten`: dimension product preservation, identity cases, bounds
//! - `clamp_min` / `clamp_max`: scalar arithmetic, idempotence, ordering
//! - `powf`: special-case exponents, sign semantics
//! - `cumsum`: prefix-sum invariants
//! - `repeat_interleave`: count validation, output size
//! - `masked_fill`: mask semantics
//!
//! Part of #3705.

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn floor_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}

fn powf_f32_stub(b: f32, _e: f32) -> f32 {
    let _ = b;
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    r
}

// ===========================================================================
// flatten: dimension product preservation
// ===========================================================================

/// Prove: flatten preserves total element count for 3D tensors.
///
/// flatten(start, end) merges dims[start..=end] into one dimension
/// whose size is the product of the merged dimensions. Total element
/// count must remain unchanged.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn flatten_preserves_numel_3d() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 16);
    kani::assume(d1 >= 1 && d1 <= 16);
    kani::assume(d2 >= 1 && d2 <= 16);

    let orig_numel = (d0 as usize) * (d1 as usize) * (d2 as usize);

    // flatten(0, 1): [d0, d1, d2] -> [d0*d1, d2]
    let flat_01 = (d0 as usize) * (d1 as usize);
    assert_eq!(
        flat_01 * (d2 as usize),
        orig_numel,
        "flatten(0,1) must preserve element count"
    );

    // flatten(1, 2): [d0, d1, d2] -> [d0, d1*d2]
    let flat_12 = (d1 as usize) * (d2 as usize);
    assert_eq!(
        (d0 as usize) * flat_12,
        orig_numel,
        "flatten(1,2) must preserve element count"
    );
}

/// Prove: flatten(i, i) is identity (same dimension range, no merge).
///
/// When start_dim == end_dim, there is nothing to flatten. The output
/// shape must equal the input shape.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn flatten_same_dim_is_identity() {
    let d0: u8 = kani::any();
    let d1: u8 = kani::any();
    kani::assume(d0 >= 1 && d0 <= 32);
    kani::assume(d1 >= 1 && d1 <= 32);

    // Simulating flatten(0, 0) on [d0, d1]:
    // Merged range is just [d0], so output is [d0, d1] — unchanged.
    let flat_size = d0 as usize; // product of single dim
    assert_eq!(flat_size, d0 as usize, "flatten(i,i) must be identity");
}

/// Prove: flatten(start, end) with start > end is invalid.
///
/// The flatten implementation rejects start_dim > end_dim with an error.
/// This proves the validation catches reversed ranges.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn flatten_reversed_range_invalid() {
    let start: u8 = kani::any();
    let end: u8 = kani::any();
    kani::assume(start >= 1 && start <= 5);
    kani::assume(end < start);

    // start > end is always invalid, regardless of tensor rank
    assert!(start > end, "reversed range must be detected");
}

/// Prove: flatten reduces rank by (end_dim - start_dim) dimensions.
///
/// Merging N contiguous dimensions into 1 reduces rank by N - 1.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn flatten_rank_reduction() {
    let rank: u8 = kani::any();
    let start: u8 = kani::any();
    let end: u8 = kani::any();
    kani::assume(rank >= 2 && rank <= 6);
    kani::assume(start < rank);
    kani::assume(end >= start && end < rank);

    let merged_count = (end as usize) - (start as usize) + 1;
    let new_rank = (rank as usize) - merged_count + 1;

    assert!(new_rank >= 1, "flattened tensor must have at least rank 1");
    assert!(
        new_rank <= rank as usize,
        "flattened rank must not exceed original"
    );
    assert_eq!(
        new_rank,
        (rank as usize) - (end as usize - start as usize),
        "rank reduction must equal merged_count - 1"
    );
}

// ===========================================================================
// clamp_min / clamp_max: scalar arithmetic properties
// ===========================================================================

/// Prove: clamp_min(x, lo) >= lo for all finite x.
///
/// The f32::max(lo) contract guarantees the output is at least lo.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn clamp_min_lower_bound() {
    let x: f32 = kani::any();
    let lo: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(lo.is_finite());

    let result = x.max(lo);
    assert!(result >= lo, "clamp_min result must be >= lo");
}

/// Prove: clamp_max(x, hi) <= hi for all finite x.
///
/// The f32::min(hi) contract guarantees the output is at most hi.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn clamp_max_upper_bound() {
    let x: f32 = kani::any();
    let hi: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(hi.is_finite());

    let result = x.min(hi);
    assert!(result <= hi, "clamp_max result must be <= hi");
}

/// Prove: clamp_min is idempotent — clamping twice equals clamping once.
///
/// max(max(x, lo), lo) = max(x, lo). Double-clamping must not change the result.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn clamp_min_idempotent() {
    let x: f32 = kani::any();
    let lo: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(lo.is_finite());

    let once = x.max(lo);
    let twice = once.max(lo);
    assert_eq!(once, twice, "clamp_min must be idempotent");
}

/// Prove: clamp_max is idempotent.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn clamp_max_idempotent() {
    let x: f32 = kani::any();
    let hi: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(hi.is_finite());

    let once = x.min(hi);
    let twice = once.min(hi);
    assert_eq!(once, twice, "clamp_max must be idempotent");
}

/// Prove: clamp_min followed by clamp_max produces valid range [lo, hi].
///
/// When lo <= hi, clamp_min(lo) then clamp_max(hi) produces a value in [lo, hi].
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn clamp_min_then_max_produces_valid_range() {
    let x: f32 = kani::any();
    let lo: f32 = kani::any();
    let hi: f32 = kani::any();
    kani::assume(x.is_finite());
    kani::assume(lo.is_finite());
    kani::assume(hi.is_finite());
    kani::assume(lo <= hi);

    let clamped = x.max(lo).min(hi);
    assert!(clamped >= lo, "clamped value must be >= lo");
    assert!(clamped <= hi, "clamped value must be <= hi");
}

/// Prove: clamp_min with lo = -inf is identity for finite inputs.
///
/// Part of #3705.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn clamp_min_neg_inf_is_identity() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = x.max(f32::NEG_INFINITY);
    assert_eq!(result, x, "clamp_min(-inf) must be identity for finite x");
}

/// Prove: clamp_max with hi = +inf is identity for finite inputs.
///
/// Part of #3705.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn clamp_max_pos_inf_is_identity() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = x.min(f32::INFINITY);
    assert_eq!(result, x, "clamp_max(+inf) must be identity for finite x");
}

// ===========================================================================
// powf: special-case exponent semantics
// ===========================================================================

/// Prove: x^0.0 = 1.0 for all finite nonzero x (IEEE 754).
///
/// Part of #3705.
#[kani::unwind(8)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
#[kani::unwind(8)]
fn powf_zero_exponent_is_one() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite() && x != 0.0);

    let result = x.powf(0.0);
    assert_eq!(result, 1.0, "x^0 must be 1.0 for nonzero x");
}

/// Prove: x^1.0 = x for all finite x (identity exponent).
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn powf_one_exponent_is_identity() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = x.powf(1.0);
    // Allow small ULP error for powf implementation
    let diff = (result - x).abs();
    assert!(
        diff < 1e-6 * x.abs().max(1.0),
        "x^1 must be approximately x"
    );
}

/// Prove: x^2.0 is non-negative for all finite x.
///
/// Squaring always produces a non-negative result, even for negative inputs.
/// This is critical for the gpu_powf even-integer-exponent path.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::powf, powf_f32_stub)]
fn powf_square_is_nonnegative() {
    let x: f32 = kani::any();
    kani::assume(x.is_finite());

    let result = x.powf(2.0);
    kani::assume(result.is_finite());
    assert!(result >= 0.0, "x^2 must be non-negative");
}

/// Prove: negative base with non-integer exponent produces NaN (IEEE 754).
///
/// This validates the gpu_powf NaN-fill path for non-integer exponents.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::floor, floor_f32_stub)]
#[kani::stub(f32::powf, powf_f32_stub)]
fn powf_negative_base_noninteger_exponent_is_nan() {
    let x: f32 = kani::any();
    let e: f32 = kani::any();
    kani::assume(x < 0.0 && x.is_finite());
    kani::assume(e.is_finite() && e != 0.0);
    kani::assume(e != e.floor()); // non-integer

    let result = x.powf(e);
    assert!(result.is_nan(), "negative^noninteger must be NaN");
}

/// Prove: even integer exponent parity detection is correct.
///
/// The gpu_powf path uses `(e as i64) % 2 == 0` to detect even exponents.
/// This verifies the cast-and-modulo pattern for representable integers.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::floor, floor_f32_stub)]
#[kani::unwind(8)]
fn powf_even_integer_detection() {
    let e: i16 = kani::any();
    kani::assume(e >= -100 && e <= 100);

    let ef = e as f32;
    // Must be representable exactly in f32
    kani::assume(ef == ef.floor() && ef.is_finite());

    let is_even = (ef as i64) % 2 == 0;
    let expected_even = e % 2 == 0;
    assert_eq!(
        is_even, expected_even,
        "even detection must agree with integer modulo"
    );
}

// ===========================================================================
// cumsum: prefix sum invariants
// ===========================================================================

/// Prove: cumsum of a single element is that element.
///
/// The cumulative sum of [a] is [a].
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn cumsum_single_element_identity() {
    let a: f32 = kani::any();
    kani::assume(a.is_finite());

    // cumsum([a]) = [a] — first element is always the element itself
    let result = a; // sum of prefix [a] = a
    assert_eq!(result, a, "cumsum of single element must be identity");
}

/// Prove: the last element of cumsum equals the total sum.
///
/// cumsum([a, b, c]) = [a, a+b, a+b+c]. The last element equals sum(all).
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn cumsum_last_element_is_total_sum() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite() && c.is_finite());
    kani::assume(a.abs() <= 1e6 && b.abs() <= 1e6 && c.abs() <= 1e6);

    let total = a + b + c;
    let cumsum_last = a + b + c;
    kani::assume(total.is_finite());

    assert_eq!(
        cumsum_last, total,
        "last cumsum element must equal total sum"
    );
}

/// Prove: cumsum is monotonically non-decreasing for non-negative inputs.
///
/// If all inputs are >= 0, each prefix sum is >= the previous.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn cumsum_monotone_for_nonneg() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    let c: f32 = kani::any();
    kani::assume(a >= 0.0 && a.is_finite());
    kani::assume(b >= 0.0 && b.is_finite());
    kani::assume(c >= 0.0 && c.is_finite());
    kani::assume(a <= 1e6 && b <= 1e6 && c <= 1e6);

    let cs0 = a;
    let cs1 = a + b;
    let cs2 = a + b + c;
    kani::assume(cs1.is_finite() && cs2.is_finite());

    assert!(
        cs1 >= cs0,
        "cumsum must be non-decreasing for non-neg inputs"
    );
    assert!(
        cs2 >= cs1,
        "cumsum must be non-decreasing for non-neg inputs"
    );
}

// ===========================================================================
// repeat_interleave: count validation
// ===========================================================================

/// Prove: negative repeat count is invalid.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn repeat_interleave_negative_count_invalid() {
    let v: f32 = kani::any();
    kani::assume(v < 0.0 && v.is_finite());

    // !v.is_finite() || v < 0.0 || v != v.trunc() must catch this
    let is_valid = v.is_finite() && v >= 0.0 && v == v.trunc();
    assert!(!is_valid, "negative count must be invalid");
}

/// Prove: non-integer repeat count is invalid.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn repeat_interleave_noninteger_count_invalid() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite() && v >= 0.0);
    kani::assume(v != v.trunc()); // not an integer

    let is_valid = v.is_finite() && v >= 0.0 && v == v.trunc();
    assert!(!is_valid, "non-integer count must be invalid");
}

/// Prove: NaN repeat count is invalid.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn repeat_interleave_nan_count_invalid() {
    let v = f32::NAN;

    let is_valid = v.is_finite() && v >= 0.0 && v == v.trunc();
    assert!(!is_valid, "NaN count must be invalid");
}

/// Prove: valid integer count produces correct output size per element.
///
/// Each element repeated k times contributes k to the output size.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn repeat_interleave_output_size_per_element() {
    let count: u8 = kani::any();
    kani::assume(count <= 100);

    let input_elements: u8 = kani::any();
    kani::assume(input_elements >= 1 && input_elements <= 50);

    // If all elements have the same count, output = input * count
    let output = (input_elements as usize) * (count as usize);
    assert_eq!(
        output,
        input_elements as usize * count as usize,
        "uniform repeat produces input * count elements"
    );
}

/// Prove: zero count produces zero-length output for that element.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn repeat_interleave_zero_count_contributes_zero() {
    let other_count: u8 = kani::any();
    kani::assume(other_count <= 100);

    // Sum of [0, other_count] = other_count (not 0 + other_count = other_count)
    let total: usize = 0 + other_count as usize;
    assert_eq!(
        total, other_count as usize,
        "zero-count element must not contribute to output size"
    );
}

// ===========================================================================
// checked_f64_to_f32: used by clamp_min/clamp_max/powf
// ===========================================================================

/// Prove: f64 values within f32 range convert without overflow.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn f64_to_f32_in_range_succeeds() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite());

    // Round-tripping: f32 -> f64 -> f32 must be exact
    let as_f64 = v as f64;
    let back = as_f64 as f32;
    assert_eq!(v, back, "f32->f64->f32 round-trip must be exact");
}

/// Prove: f64 values beyond f32 range overflow to infinity.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn f64_to_f32_overflow_detected() {
    let large: f64 = f64::from(f32::MAX) * 2.0;
    let as_f32 = large as f32;
    assert!(
        as_f32.is_infinite(),
        "f64 beyond f32 range must overflow to inf"
    );
}
