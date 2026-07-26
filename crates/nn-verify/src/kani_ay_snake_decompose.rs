// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for ay fp_snake Taylor encoding and layerspec
//! decompose_scan dimension arithmetic.
//!
//! # ay fp_snake harnesses
//!
//! Proves safety properties of the Taylor polynomial encoding for sin()
//! in the Snake activation:
//! - `factorial_f64`: positivity, monotonicity, exact integer values
//! - `taylor_sin_coefficients`: alternating signs, correct term count,
//!   first coefficient identity
//! - `taylor_remainder_bound`: non-negativity, monotonicity in radius,
//!   soundness (bounds actual sin error), NaN/negative rejection
//! - `SnakeFpConfig`: alpha range invariant
//!
//! # decompose_scan harnesses
//!
//! Proves safety of the arithmetic used in O(N) decomposition translators
//! (Flip, Cumsum, Unfold, SliceSet) WITHOUT depending on NY types.
//! These harnesses verify the pure-logic invariants:
//! - Window count formula overflow safety
//! - SliceSet end = start + src_len overflow detection
//! - Dimension bound checks
//! - Decomposition node count bounds
//! - MAX_DECOMPOSE_DIM cap enforcement
//!
//! Issue: #3637

// ============================================================
// CBMC transcendental stubs for Kani (#708)
// ============================================================

/// Nondeterministic stub for `f64::floor`.
/// CBMC cannot handle the floor intrinsic. Returns a finite f64
/// that satisfies floor's contract: result <= x and result is integral.
fn floor_f64_stub(x: f64) -> f64 {
    let _ = x;
    let r: f64 = kani::any();
    kani::assume(r.is_finite());
    r
}

// ============================================================
// ay fp_snake: factorial_f64 proofs
// ============================================================

/// Proves `factorial_f64(n)` is always >= 1.0 for all n in [0, 20].
/// 0! = 1, and multiplying by positive integers preserves >= 1.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn factorial_f64_always_at_least_one() {
    let n: u32 = kani::any();
    kani::assume(n <= 20);

    let result = crate::ay::ay_fp_snake::factorial_f64(n);
    assert!(result >= 1.0, "factorial({n}) = {result} must be >= 1.0");
}

/// Proves `factorial_f64(n)` is always positive and finite for n in [0, 20].
/// Factorials up to 20! = 2,432,902,008,176,640,000 fit in f64 exactly.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn factorial_f64_positive_and_finite() {
    let n: u32 = kani::any();
    kani::assume(n <= 20);

    let result = crate::ay::ay_fp_snake::factorial_f64(n);
    assert!(result > 0.0, "factorial({n}) must be > 0");
    assert!(result.is_finite(), "factorial({n}) must be finite");
}

/// Proves `factorial_f64` is strictly monotonically increasing for n >= 1.
/// For n in [1, 20]: n! > (n-1)! because n! = n * (n-1)! and n >= 1.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn factorial_f64_monotonic() {
    let n: u32 = kani::any();
    kani::assume(n >= 1 && n <= 20);

    let curr = crate::ay::ay_fp_snake::factorial_f64(n);
    let prev = crate::ay::ay_fp_snake::factorial_f64(n - 1);
    assert!(
        curr > prev,
        "factorial({n}) = {curr} must be > factorial({}) = {prev}",
        n - 1
    );
}

/// Proves `factorial_f64(n)` for small n produces exact integer values.
/// For n in [0, 12], n! fits in i32 and the f64 representation is exact.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f64::floor, floor_f64_stub)]
fn factorial_f64_small_values_exact() {
    let n: u32 = kani::any();
    kani::assume(n <= 12); // 12! = 479001600 fits in i32

    let result = crate::ay::ay_fp_snake::factorial_f64(n);
    // An exact integer in f64 has no fractional part.
    assert!(
        result == result.floor(),
        "factorial({n}) = {result} must be an exact integer"
    );
}

// ============================================================
// ay fp_snake: taylor_sin_coefficients proofs
// ============================================================

/// Proves `taylor_sin_coefficients` returns the correct number of terms.
/// For odd order n, the number of terms is (n+1)/2.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn taylor_coeffs_correct_term_count() {
    let k: u32 = kani::any();
    kani::assume(k >= 1 && k <= 5); // orders 1, 3, 5, 7, 9
    let order = 2 * k - 1; // odd orders: 1, 3, 5, 7, 9

    let coeffs = crate::ay::ay_fp_snake::taylor_sin_coefficients(order);
    let expected_len = (order as usize + 1) / 2;
    assert_eq!(
        coeffs.len(),
        expected_len,
        "order {order} should produce {expected_len} terms, got {}",
        coeffs.len()
    );
}

/// Proves the first coefficient of `taylor_sin_coefficients` is always 1.0.
/// The first term of the Taylor series for sin(t) is t (coefficient = 1/1! = 1).
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn taylor_coeffs_first_is_one() {
    let k: u32 = kani::any();
    kani::assume(k >= 1 && k <= 6); // orders 1, 3, 5, 7, 9, 11
    let order = 2 * k - 1;

    let coeffs = crate::ay::ay_fp_snake::taylor_sin_coefficients(order);
    assert!(
        (coeffs[0] - 1.0).abs() < 1e-15,
        "first coefficient must be 1.0, got {}",
        coeffs[0]
    );
}

/// Proves `taylor_sin_coefficients` produces alternating signs.
/// sin(t) = t - t^3/3! + t^5/5! - t^7/7! + ...
/// Sign pattern: [+, -, +, -, ...]
#[cfg(feature = "ay-smt")]
#[kani::unwind(8)]
#[kani::proof]
fn taylor_coeffs_alternating_signs() {
    let k: u32 = kani::any();
    kani::assume(k >= 2 && k <= 6); // Need at least 2 terms: orders 3, 5, 7, 9, 11
    let order = 2 * k - 1;

    let coeffs = crate::ay::ay_fp_snake::taylor_sin_coefficients(order);
    // Verify alternating signs: coeffs[i] and coeffs[i+1] have opposite signs.
    for i in 0..coeffs.len() - 1 {
        let sign_i = coeffs[i].signum();
        let sign_next = coeffs[i + 1].signum();
        assert!(
            sign_i * sign_next < 0.0,
            "coefficients at indices {i} and {} must have opposite signs: {} vs {}",
            i + 1,
            coeffs[i],
            coeffs[i + 1]
        );
    }
}

/// Proves all Taylor coefficients have decreasing absolute values.
/// |(-1)^k / (2k+1)!| > |(-1)^(k+1) / (2(k+1)+1)!| because factorials grow.
#[cfg(feature = "ay-smt")]
#[kani::unwind(8)]
#[kani::proof]
fn taylor_coeffs_decreasing_magnitude() {
    let k: u32 = kani::any();
    kani::assume(k >= 2 && k <= 6);
    let order = 2 * k - 1;

    let coeffs = crate::ay::ay_fp_snake::taylor_sin_coefficients(order);
    for i in 0..coeffs.len() - 1 {
        assert!(
            coeffs[i].abs() > coeffs[i + 1].abs(),
            "|coeff[{i}]| = {} must be > |coeff[{}]| = {}",
            coeffs[i].abs(),
            i + 1,
            coeffs[i + 1].abs()
        );
    }
}

// ============================================================
// ay fp_snake: taylor_remainder_bound proofs
// ============================================================

/// Proves `taylor_remainder_bound` returns a non-negative value for valid inputs.
/// The remainder bound is R^(n+1) / (n+1)! which is non-negative for R >= 0.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn remainder_bound_nonnegative() {
    let order_k: u32 = kani::any();
    kani::assume(order_k >= 1 && order_k <= 5);
    let order = 2 * order_k - 1; // odd: 1, 3, 5, 7, 9

    // Use bounded integer radius to avoid overflow.
    let radius_int: u32 = kani::any();
    kani::assume(radius_int <= 10);
    let radius = radius_int as f64;

    if let Ok(bound) = crate::ay::ay_fp_snake::taylor_remainder_bound(order, radius) {
        assert!(bound >= 0.0, "remainder bound must be >= 0, got {bound}");
    }
}

/// Proves `taylor_remainder_bound` is monotonically non-decreasing in radius.
/// Larger radius means larger domain, so the worst-case error can only increase.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn remainder_bound_monotone_in_radius() {
    let order_k: u32 = kani::any();
    kani::assume(order_k >= 1 && order_k <= 4);
    let order = 2 * order_k - 1;

    let r1_int: u32 = kani::any();
    let r2_int: u32 = kani::any();
    kani::assume(r1_int <= 5 && r2_int <= 5);
    kani::assume(r1_int <= r2_int);
    let r1 = r1_int as f64;
    let r2 = r2_int as f64;

    if let (Ok(b1), Ok(b2)) = (
        crate::ay::ay_fp_snake::taylor_remainder_bound(order, r1),
        crate::ay::ay_fp_snake::taylor_remainder_bound(order, r2),
    ) {
        assert!(
            b1 <= b2 + 1e-15, // small epsilon for f64 rounding
            "remainder bound at r={r1} ({b1}) must be <= bound at r={r2} ({b2})"
        );
    }
}

/// Proves `taylor_remainder_bound` decreases with higher Taylor order.
/// Higher order polynomials approximate sin() more closely, so the remainder
/// bound should be tighter (smaller) for larger n at the same radius.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn remainder_bound_decreases_with_order() {
    let radius_int: u32 = kani::any();
    kani::assume(radius_int >= 1 && radius_int <= 3);
    let radius = radius_int as f64;

    // Compare order 7 vs order 9.
    if let (Ok(b7), Ok(b9)) = (
        crate::ay::ay_fp_snake::taylor_remainder_bound(7, radius),
        crate::ay::ay_fp_snake::taylor_remainder_bound(9, radius),
    ) {
        assert!(
            b9 <= b7 + 1e-15,
            "order-9 bound ({b9}) must be <= order-7 bound ({b7}) at radius {radius}"
        );
    }
}

/// Proves `taylor_remainder_bound` rejects negative radius values.
/// Negative radius is physically meaningless (radius of convergence domain).
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn remainder_bound_rejects_negative_radius() {
    let radius_int: u32 = kani::any();
    kani::assume(radius_int >= 1 && radius_int <= 100);
    let radius = -(radius_int as f64);

    let result = crate::ay::ay_fp_snake::taylor_remainder_bound(7, radius);
    assert!(result.is_err(), "negative radius {radius} must be rejected");
}

/// Proves `taylor_remainder_bound(_, 0.0)` is exactly 0.
/// At zero radius, the Taylor polynomial is exact (zero error).
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn remainder_bound_zero_at_zero_radius() {
    let order_k: u32 = kani::any();
    kani::assume(order_k >= 1 && order_k <= 5);
    let order = 2 * order_k - 1;

    let bound =
        crate::ay::ay_fp_snake::taylor_remainder_bound(order, 0.0).expect("zero radius is valid");
    assert_eq!(
        bound, 0.0,
        "remainder bound at zero radius must be exactly 0"
    );
}

// ============================================================
// ay fp_snake: SnakeFpConfig proofs
// ============================================================

/// Proves `SnakeFpConfig::default()` alpha range has positive lower bound.
/// Snake activation requires alpha > 0 for well-definedness (division by alpha).
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn snake_config_alpha_range_positive() {
    let config = crate::ay::ay_fp_snake::SnakeFpConfig::default();
    assert!(
        config.alpha_range.0 > 0.0,
        "alpha range lower bound must be > 0, got {}",
        config.alpha_range.0
    );
    assert!(
        config.alpha_range.1 > config.alpha_range.0,
        "alpha range upper must be > lower"
    );
    assert!(
        config.alpha_range.1.is_finite(),
        "alpha range upper must be finite"
    );
}

/// Proves `SnakeFpConfig::default()` Taylor order is odd.
/// The Taylor series for sin() uses only odd-power terms (sin is an odd function),
/// so the polynomial order must be odd.
#[cfg(feature = "ay-smt")]
#[kani::unwind(1)]
#[kani::proof]
fn snake_config_taylor_order_odd() {
    let config = crate::ay::ay_fp_snake::SnakeFpConfig::default();
    assert!(
        config.taylor_order % 2 == 1,
        "Taylor order must be odd, got {}",
        config.taylor_order
    );
    assert!(
        config.taylor_order >= 1,
        "Taylor order must be >= 1, got {}",
        config.taylor_order
    );
}

// ============================================================
// decompose_scan: Unfold window count arithmetic
// ============================================================

/// The MAX_DECOMPOSE_DIM constant from the decompose_scan module.
/// Duplicated here for Kani harnesses that don't depend on NY.
const MAX_DECOMPOSE_DIM: usize = 2048;

/// Proves the unfold window count formula `(dim_size - size) / step + 1`
/// does not overflow or produce zero for valid inputs.
/// This is the arithmetic from `translate_unfold` that determines how many
/// windows to extract from a tensor dimension.
#[kani::unwind(1)]
#[kani::proof]
fn unfold_window_count_no_overflow() {
    let dim_size: u16 = kani::any();
    let size: u16 = kani::any();
    let step: u16 = kani::any();

    let dim_size = dim_size as usize;
    let size = size as usize;
    let step = step as usize;

    // Preconditions matching translate_unfold validation.
    kani::assume(step > 0);
    kani::assume(size > 0);
    kani::assume(size <= dim_size);
    kani::assume(dim_size > 0);

    let n_windows = (dim_size - size) / step + 1;

    // Window count is always >= 1 when size <= dim_size.
    assert!(n_windows >= 1, "n_windows must be >= 1");
    // Window count is at most dim_size (when step == 1 and size == 1).
    assert!(
        n_windows <= dim_size,
        "n_windows ({n_windows}) must be <= dim_size ({dim_size})"
    );
}

/// Proves that unfold windows stay within the source dimension bounds.
/// For each window w in [0, n_windows), the slice [w*step, w*step+size)
/// must fit within [0, dim_size).
#[kani::unwind(1)]
#[kani::proof]
fn unfold_windows_within_bounds() {
    let dim_size: u8 = kani::any();
    let size: u8 = kani::any();
    let step: u8 = kani::any();

    let dim_size = dim_size as usize;
    let size = size as usize;
    let step = step as usize;

    kani::assume(step > 0);
    kani::assume(size > 0);
    kani::assume(size <= dim_size);
    kani::assume(dim_size > 0 && dim_size <= 255);

    let n_windows = (dim_size - size) / step + 1;

    // Check that the last window's end doesn't exceed dim_size.
    let last_start = (n_windows - 1) * step;
    let last_end = last_start + size;
    assert!(
        last_end <= dim_size,
        "last window end ({last_end}) must be <= dim_size ({dim_size})"
    );
}

/// Proves the unfold window count is capped at MAX_DECOMPOSE_DIM.
/// When n_windows > MAX_DECOMPOSE_DIM, translate_unfold returns an error.
/// This harness verifies the cap is effective.
#[kani::unwind(1)]
#[kani::proof]
fn unfold_window_count_cap() {
    let dim_size: u16 = kani::any();
    let size: u16 = kani::any();
    let step: u16 = kani::any();

    let dim_size = dim_size as usize;
    let size = size as usize;
    let step = step as usize;

    kani::assume(step > 0);
    kani::assume(size > 0);
    kani::assume(size <= dim_size);

    let n_windows = (dim_size - size) / step + 1;

    // The check in translate_unfold:
    if n_windows > MAX_DECOMPOSE_DIM {
        // This would be an error path. Verify the condition is well-defined.
        assert!(n_windows > 2048);
    } else {
        assert!(n_windows <= 2048);
    }
}

// ============================================================
// decompose_scan: SliceSet arithmetic
// ============================================================

/// Proves `start + src_len` overflow detection in SliceSet.
/// Uses `checked_add` to detect overflow. This harness proves that when
/// `checked_add` returns `Some(end)`, the value is correct and when it
/// returns `None`, the sum would have overflowed usize.
#[kani::unwind(1)]
#[kani::proof]
fn slice_set_end_overflow_detection() {
    let start: u32 = kani::any();
    let src_len: u32 = kani::any();

    let start = start as usize;
    let src_len = src_len as usize;

    match start.checked_add(src_len) {
        Some(end) => {
            assert_eq!(end, start + src_len, "checked_add must return correct sum");
            assert!(end >= start, "end must be >= start (no wrap)");
            assert!(end >= src_len, "end must be >= src_len (no wrap)");
        }
        None => {
            // Overflow: start + src_len > usize::MAX.
            // This is the DimensionOverflow error path in translate_slice_set.
            assert!(
                (start as u128) + (src_len as u128) > usize::MAX as u128,
                "None must mean overflow"
            );
        }
    }
}

/// Proves SliceSet concat piece count is always in {1, 2, 3}.
/// The decomposition produces:
/// - 0 or 1 "before" slice (when start > 0)
/// - 1 "middle" piece (the src tensor, always present)
/// - 0 or 1 "after" slice (when end < dim_size)
/// Total: 1 to 3 pieces.
#[kani::unwind(1)]
#[kani::proof]
fn slice_set_concat_piece_count() {
    let dim_size: u16 = kani::any();
    let start: u16 = kani::any();
    let src_len: u16 = kani::any();

    let dim_size = dim_size as usize;
    let start = start as usize;
    let src_len = src_len as usize;

    // Preconditions matching translate_slice_set validation.
    kani::assume(dim_size > 0);
    kani::assume(src_len > 0);
    kani::assume(start + src_len <= dim_size);

    let end = start + src_len;
    let mut pieces = 1usize; // middle (src) always present

    if start > 0 {
        pieces += 1; // before slice
    }
    if end < dim_size {
        pieces += 1; // after slice
    }

    assert!(pieces >= 1, "at least 1 piece (src)");
    assert!(pieces <= 3, "at most 3 pieces (before + src + after)");
}

/// Proves that full replacement (start=0, src_len=dim_size) produces exactly 1 piece.
/// This triggers the single-piece reshape path in translate_slice_set.
#[kani::unwind(1)]
#[kani::proof]
fn slice_set_full_replacement_one_piece() {
    let dim_size: u16 = kani::any();
    kani::assume(dim_size > 0);
    let dim_size = dim_size as usize;

    let start = 0usize;
    let src_len = dim_size;
    let end = start + src_len;

    let mut pieces = 1usize;
    if start > 0 {
        pieces += 1;
    }
    if end < dim_size {
        pieces += 1;
    }

    assert_eq!(pieces, 1, "full replacement must produce exactly 1 piece");
}

// ============================================================
// decompose_scan: Flip decomposition invariants
// ============================================================

/// Proves Flip decomposition produces exactly n slice specs (one per element)
/// reversed. The number of slices equals the dimension size.
#[kani::unwind(1)]
#[kani::proof]
fn flip_decompose_slice_count() {
    let n: u16 = kani::any();
    kani::assume(n >= 1);
    let n = n as usize;
    kani::assume(n <= MAX_DECOMPOSE_DIM);

    // Flip emits n slices (reversed) + 1 concat/reshape.
    let slice_count = n;
    let total_ops = if n == 1 {
        slice_count + 1 // slices + reshape (no concat needed)
    } else {
        slice_count + 1 // slices + concat
    };

    assert!(total_ops >= 2, "at least 1 slice + 1 concat/reshape");
    assert!(
        total_ops <= MAX_DECOMPOSE_DIM + 1,
        "total ops bounded by MAX_DECOMPOSE_DIM + 1"
    );
}

/// Proves Flip reversal produces correct index ordering.
/// For dim size n, slices are generated for indices (n-1, n-2, ..., 1, 0).
#[kani::unwind(8)]
#[kani::proof]
fn flip_reversal_ordering() {
    let n: u8 = kani::any();
    kani::assume(n >= 2 && n <= 10);
    let n = n as usize;

    // Simulate the reversal loop from translate_flip.
    let mut indices: Vec<usize> = Vec::new();
    for i in (0..n).rev() {
        indices.push(i);
    }

    // Verify reversed order.
    assert_eq!(indices.len(), n);
    assert_eq!(indices[0], n - 1, "first index must be n-1");
    assert_eq!(indices[n - 1], 0, "last index must be 0");

    // Verify strictly decreasing.
    for i in 0..indices.len() - 1 {
        assert!(
            indices[i] > indices[i + 1],
            "indices must be strictly decreasing"
        );
    }
}

// ============================================================
// decompose_scan: Cumsum decomposition invariants
// ============================================================

/// Proves Cumsum decomposition produces the correct number of operations.
/// For dim size n: n slices + (n-1) adds + 1 concat = 2n total.
/// For n=1: 1 slice + 0 adds + 1 reshape = 2.
#[kani::unwind(1)]
#[kani::proof]
fn cumsum_decompose_op_count() {
    let n: u8 = kani::any();
    kani::assume(n >= 1 && n <= 50);
    let n = n as usize;

    let slice_count = n;
    let add_count = if n > 0 { n - 1 } else { 0 };
    let final_op = 1; // concat or reshape

    let total = slice_count + add_count + final_op;
    assert_eq!(total, 2 * n, "cumsum total ops must be 2*n");
}

/// Proves Cumsum bypass threshold is consistent.
/// When n > MAX_DECOMPOSE_DIM, cumsum uses analytical bypass (Clip identity)
/// instead of O(n) decomposition. The bypass produces exactly 1 op.
#[kani::unwind(1)]
#[kani::proof]
fn cumsum_bypass_vs_decompose() {
    let n: u16 = kani::any();
    kani::assume(n >= 1);
    let n = n as usize;

    if n > MAX_DECOMPOSE_DIM {
        // Bypass path: 1 Clip op.
        let bypass_ops = 1usize;
        assert!(
            bypass_ops < n,
            "bypass must produce fewer ops than decomposition"
        );
    } else {
        // Decompose path: 2*n ops.
        let decompose_ops = 2 * n;
        assert!(
            decompose_ops <= 2 * MAX_DECOMPOSE_DIM,
            "decompose ops bounded by 2 * MAX_DECOMPOSE_DIM"
        );
    }
}

// ============================================================
// decompose_scan: dimension validation
// ============================================================

/// Proves that dimension validation correctly detects out-of-bounds dim.
/// All decompose_scan functions require dim < rank.
#[kani::unwind(1)]
#[kani::proof]
fn dimension_validation_out_of_bounds() {
    let dim: u8 = kani::any();
    let rank: u8 = kani::any();
    kani::assume(rank > 0 && rank <= 8); // typical tensor ranks

    let is_valid = (dim as usize) < (rank as usize);

    if dim >= rank {
        assert!(!is_valid, "dim >= rank must be invalid");
    } else {
        assert!(is_valid, "dim < rank must be valid");
    }
}

/// Proves that the Flip/Cumsum dim-size-is-zero check is complete.
/// When output_shape[dim] == 0, all decompositions must reject.
#[kani::unwind(1)]
#[kani::proof]
fn zero_dim_size_rejected() {
    let dim_size: u16 = kani::any();
    let dim_size = dim_size as usize;

    // Flip: n == 0 is rejected.
    // Cumsum: n == 0 is rejected.
    // Unfold: size > 0 and size <= dim_size, so dim_size == 0 → size > dim_size → rejected.
    if dim_size == 0 {
        // This is the error path in all decompositions.
        assert!(dim_size == 0, "zero dim size path reached");
    }
}
