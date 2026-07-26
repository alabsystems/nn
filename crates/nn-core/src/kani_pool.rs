// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for pool2d output-length arithmetic and
//! adaptive average pooling window-index safety.
//!
//! Verifies that the arithmetic in `pool2d_out_len` (dyn_tensor/conv/pool.rs)
//! and adaptive_avg_pool2d window indexing cannot produce incorrect results
//! through integer overflow or logic errors.
//!
//! Harnesses inline the arithmetic expressions rather than importing private
//! functions. This proves the arithmetic properties independent of
//! error-handling paths.

#![cfg(kani)]

// ---------------------------------------------------------------------------
// pool2d_out_len arithmetic harnesses
// ---------------------------------------------------------------------------

/// Prove: pool2d_out_len arithmetic does not panic for small valid params.
///
/// Inlines: `padded = il + 2 * p; out = (padded - ks) / s + 1`
///
/// Uses u8 inputs to cover the full small parameter space exhaustively.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_no_panic_small() {
    let il: u8 = kani::any(); // input_len
    let ks: u8 = kani::any(); // kernel_size
    let p: u8 = kani::any(); // padding
    let s: u8 = kani::any(); // stride

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;

    // Match dyn_tensor/conv/pool.rs:36-46
    let padded = il + 2 * p;
    if padded >= ks {
        let numerator = padded - ks;
        let out = numerator / s + 1;
        assert!(out >= 1, "pool2d output length must be >= 1 when valid");
    }
}

/// Prove: pool2d identity pool (k=1, s=1, p=0) preserves input length.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_identity() {
    let il: u8 = kani::any();
    kani::assume(il >= 1);

    let padded = il as usize + 0; // p=0
    let numerator = padded - 1; // ks=1
    let out = numerator / 1 + 1; // s=1
    assert_eq!(out, il as usize);
}

/// Prove: pool2d output is monotonically non-increasing in stride.
///
/// Larger stride skips more elements, producing fewer output positions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_monotone_decreasing_in_stride() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s1: u8 = kani::any();
    let s2: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s1 >= 1);
    kani::assume(s2 >= s1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;

    let padded = il + 2 * p;
    if padded >= ks {
        let numerator = padded - ks;
        let o1 = numerator / (s1 as usize) + 1;
        let o2 = numerator / (s2 as usize) + 1;
        assert!(o1 >= o2, "larger stride must produce <= output");
    }
}

// ---------------------------------------------------------------------------
// pool2d_out_len ceil_mode harnesses
// ---------------------------------------------------------------------------

/// Prove: ceil_mode output >= floor mode output for all valid params.
///
/// floor: `(padded - ks) / s + 1`
/// ceil:  `(padded - ks).div_ceil(s) + 1`
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_ceil_gte_floor() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;

    let padded = il + 2 * p;
    if padded >= ks {
        let numerator = padded - ks;
        let floor_out = numerator / s + 1;
        // div_ceil for usize: (a + b - 1) / b
        let ceil_out = (numerator + s - 1) / s + 1;
        assert!(
            ceil_out >= floor_out,
            "ceil_mode must produce >= floor_mode output"
        );
    }
}

/// Prove: ceil_mode and floor_mode agree when numerator is divisible by stride.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_ceil_eq_floor_when_divisible() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;

    let padded = il + 2 * p;
    if padded >= ks {
        let numerator = padded - ks;
        kani::assume(numerator % s == 0); // exactly divisible
        let floor_out = numerator / s + 1;
        let ceil_out = (numerator + s - 1) / s + 1;
        assert_eq!(
            ceil_out, floor_out,
            "ceil and floor must agree when exactly divisible"
        );
    }
}

/// Prove: ceil_mode output is at most 1 more than floor_mode.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_ceil_exceeds_floor_by_at_most_one() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;

    let padded = il + 2 * p;
    if padded >= ks {
        let numerator = padded - ks;
        let floor_out = numerator / s + 1;
        let ceil_out = (numerator + s - 1) / s + 1;
        assert!(
            ceil_out <= floor_out + 1,
            "ceil_mode exceeds floor_mode by at most 1"
        );
    }
}

// ---------------------------------------------------------------------------
// adaptive_avg_pool2d window-index safety harnesses
// ---------------------------------------------------------------------------

/// Prove: adaptive pooling window indices are within bounds.
///
/// Verifies that for all valid (oh, out_h, in_h) triples:
/// - start_h = (oh * in_h) / out_h < in_h
/// - end_h = ((oh + 1) * in_h).div_ceil(out_h) <= in_h
/// - start_h < end_h (windows are always non-empty due to div_ceil)
///
/// Production code (pool.rs:308-309) uses floor for start_h and
/// div_ceil for end_h. The ceiling division ensures every output
/// position maps to at least one input element, even when out_h > in_h.
///
/// Uses u8 to cover small parameter space exhaustively.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn adaptive_pool_window_bounds() {
    let oh: u8 = kani::any(); // output index
    let out_h: u8 = kani::any(); // output size
    let in_h: u8 = kani::any(); // input size

    kani::assume(out_h >= 1);
    kani::assume(in_h >= 1);
    kani::assume(oh < out_h);

    let oh = oh as usize;
    let out_h = out_h as usize;
    let in_h = in_h as usize;

    // Match dyn_tensor/conv/pool.rs:308-309 exactly
    let start_h = (oh * in_h) / out_h;
    let end_h = ((oh + 1) * in_h).div_ceil(out_h);

    assert!(start_h < in_h, "start index must be within input bounds");
    assert!(end_h <= in_h, "end index must not exceed input size");
    // div_ceil guarantees non-empty windows for all out_h/in_h ratios
    assert!(
        start_h < end_h,
        "window must be non-empty (div_ceil guarantees this)"
    );
}

/// Prove: adaptive pooling windows are non-empty for all out_h/in_h ratios.
///
/// Production code (pool.rs:308-309) uses div_ceil for end_h, which
/// guarantees at least one input element per output position even when
/// upsampling (out_h > in_h). This harness verifies the non-empty
/// property holds universally — not just for downsampling.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn adaptive_pool_window_nonempty_all_ratios() {
    let oh: u8 = kani::any();
    let out_h: u8 = kani::any();
    let in_h: u8 = kani::any();

    kani::assume(out_h >= 1);
    kani::assume(in_h >= 1);
    kani::assume(oh < out_h);

    let oh = oh as usize;
    let out_h = out_h as usize;
    let in_h = in_h as usize;

    // Match dyn_tensor/conv/pool.rs:308-309 exactly
    let start_h = (oh * in_h) / out_h;
    let end_h = ((oh + 1) * in_h).div_ceil(out_h);

    assert!(
        start_h < end_h,
        "window must be non-empty (div_ceil guarantees this for all ratios)"
    );
}

/// Prove: adaptive pooling covers entire input exactly
/// (window start at oh=0 is 0, window end at oh=out_h-1 is in_h).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn adaptive_pool_covers_full_input() {
    let out_h: u8 = kani::any();
    let in_h: u8 = kani::any();

    kani::assume(out_h >= 1);
    kani::assume(in_h >= 1);

    let out_h = out_h as usize;
    let in_h = in_h as usize;

    // First window starts at 0
    let first_start = (0_usize * in_h) / out_h;
    assert_eq!(first_start, 0, "first window must start at 0");

    // Last window ends at in_h
    let last_end = (out_h * in_h) / out_h;
    assert_eq!(last_end, in_h, "last window must end at in_h");
}
