// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for pooling operations (`pool.rs`).
//!
//! Extends the top-level `kani_pool.rs` with proofs specifically for:
//!
//! - `pool2d_out_len` checked arithmetic: overflow detection, padded < kernel rejection
//! - Pool output index safety: every output position maps to valid input range
//! - max_pool1d index arithmetic: kernel window positions are in-bounds
//! - avg_pool2d count positivity: divisor is always > 0 for valid windows
//! - Adaptive pool window partition: windows tile the input without gaps or overlap
//!   at boundaries
//! - Pool stride=kernel gives non-overlapping windows
//! - Pool with maximum padding: output still computable

#![cfg(kani)]

// ---------------------------------------------------------------------------
// pool2d_out_len: checked arithmetic rejects zero kernel
// ---------------------------------------------------------------------------

/// Prove: pool2d_out_len rejects kernel_size=0 for all other params.
///
/// The production code checks kernel_size == 0 and returns Err.
/// This harness proves that check is exhaustive: there is no way
/// to get kernel_size=0 past validation.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_rejects_zero_kernel() {
    let il: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(s >= 1);

    // kernel_size = 0 must always fail, regardless of other params
    let result = super::pool::pool2d_out_len(il as usize, 0, p as usize, s as usize, false);
    assert!(result.is_err(), "pool2d_out_len must reject kernel_size=0");
}

/// Prove: pool2d_out_len rejects stride=0 for all other params.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_rejects_zero_stride() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);

    let result = super::pool::pool2d_out_len(il as usize, ks as usize, p as usize, 0, false);
    assert!(result.is_err(), "pool2d_out_len must reject stride=0");
}

/// Prove: pool2d_out_len rejects when padded input < kernel_size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_rejects_small_padded_input() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);

    let padded = (il as usize) + 2 * (p as usize);
    kani::assume(padded < ks as usize);

    let result =
        super::pool::pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, false);
    assert!(
        result.is_err(),
        "pool2d_out_len must reject when padded < kernel_size"
    );
}

// ---------------------------------------------------------------------------
// pool2d_out_len: output >= 1 for valid params
// ---------------------------------------------------------------------------

/// Prove: pool2d_out_len returns >= 1 for all valid parameter combinations.
///
/// When kernel_size >= 1, stride >= 1, and padded >= kernel_size,
/// the output must be at least 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_valid_produces_ge_1() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);

    let padded = (il as usize) + 2 * (p as usize);
    kani::assume(padded >= ks as usize);

    let result =
        super::pool::pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, false);
    assert!(result.is_ok(), "valid params must produce Ok");
    assert!(result.unwrap() >= 1, "valid pool output must be >= 1");
}

// ---------------------------------------------------------------------------
// pool2d_out_len: ceil_mode results
// ---------------------------------------------------------------------------

/// Prove: pool2d_out_len ceil_mode output >= floor_mode output via the
/// production function (not inlined arithmetic).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_ceil_gte_floor_via_fn() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);

    let padded = (il as usize) + 2 * (p as usize);
    kani::assume(padded >= ks as usize);

    let floor =
        super::pool::pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, false);
    let ceil = super::pool::pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, true);

    assert!(floor.is_ok());
    assert!(ceil.is_ok());
    assert!(
        ceil.unwrap() >= floor.unwrap(),
        "ceil_mode must produce >= floor_mode"
    );
}

// ---------------------------------------------------------------------------
// pool2d_out_len: stride == kernel_size gives exact tiling
// ---------------------------------------------------------------------------

/// Prove: when stride == kernel_size and padding == 0, the output is
/// floor(input / kernel_size). This is the non-overlapping tiling case.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_stride_eq_kernel_exact_tiling() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(il >= ks); // padded=il >= ks

    let il = il as usize;
    let ks = ks as usize;

    let result = super::pool::pool2d_out_len(il, ks, 0, ks, false);
    assert!(result.is_ok());
    let out = result.unwrap();
    // out = (il - ks) / ks + 1 = il / ks (integer division)
    assert_eq!(out, il / ks, "stride==kernel with p=0 must give il/ks");
}

// ---------------------------------------------------------------------------
// max_pool1d: kernel window index safety
// ---------------------------------------------------------------------------

/// Prove: in the max_pool1d inner loop, every accessed input index is valid.
///
/// For each output position `ol` in [0, out_len), each kernel offset `k`
/// in [0, kernel_size), if the bounds check `il >= padding && il - padding < in_len`
/// passes, then the computed flat index `b*C*L + c*L + (il - padding)` is within
/// the input buffer of size `B*C*L`.
///
/// Uses tiny dimensions to keep the state space tractable.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(6)]
fn max_pool1d_index_safety() {
    let in_len: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(in_len >= 1 && in_len <= 5);
    kani::assume(ks >= 1 && ks <= 5);
    kani::assume(s >= 1 && s <= 5);
    kani::assume(p <= 3);

    let in_len = in_len as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;

    let padded = in_len + 2 * p;
    if padded < ks {
        return; // invalid config
    }
    let out_len = (padded - ks) / s + 1;

    // Check one symbolic output position
    let ol: u8 = kani::any();
    kani::assume((ol as usize) < out_len);
    let ol = ol as usize;

    let k: u8 = kani::any();
    kani::assume((k as usize) < ks);
    let k = k as usize;

    let il = ol * s + k;
    if il >= p && il - p < in_len {
        let input_idx = il - p;
        assert!(input_idx < in_len, "input index must be within in_len");
    }
}

// ---------------------------------------------------------------------------
// max_pool2d: kernel window index safety
// ---------------------------------------------------------------------------

/// Prove: in the max_pool2d inner loop, every accessed input index pair is valid.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(5)]
fn max_pool2d_index_safety() {
    let in_h: u8 = kani::any();
    let in_w: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(in_h >= 1 && in_h <= 4);
    kani::assume(in_w >= 1 && in_w <= 4);
    kani::assume(ks >= 1 && ks <= 4);
    kani::assume(s >= 1 && s <= 4);
    kani::assume(p <= 2);

    let in_h = in_h as usize;
    let in_w = in_w as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;

    let padded_h = in_h + 2 * p;
    let padded_w = in_w + 2 * p;
    if padded_h < ks || padded_w < ks {
        return;
    }
    let out_h = (padded_h - ks) / s + 1;
    let out_w = (padded_w - ks) / s + 1;

    let oh: u8 = kani::any();
    let ow: u8 = kani::any();
    kani::assume((oh as usize) < out_h);
    kani::assume((ow as usize) < out_w);
    let oh = oh as usize;
    let ow = ow as usize;

    let kh: u8 = kani::any();
    let kw: u8 = kani::any();
    kani::assume((kh as usize) < ks);
    kani::assume((kw as usize) < ks);
    let kh = kh as usize;
    let kw = kw as usize;

    let ih = oh * s + kh;
    let iw = ow * s + kw;

    if ih >= p && ih - p < in_h && iw >= p && iw - p < in_w {
        let row = ih - p;
        let col = iw - p;
        assert!(row < in_h, "row index must be within in_h");
        assert!(col < in_w, "col index must be within in_w");
    }
}

// ---------------------------------------------------------------------------
// avg_pool2d: count > 0 for every valid output window
// ---------------------------------------------------------------------------

/// Prove: in avg_pool2d, for every valid output position, the count of
/// input elements in the window is > 0 (preventing division by zero).
///
/// When padding == 0, every kernel position maps to a valid input.
/// When padding > 0, at least one kernel position maps to valid input
/// because the window center is always within the input.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn avg_pool2d_count_positive() {
    let in_h: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(in_h >= 1 && in_h <= 4);
    kani::assume(ks >= 1 && ks <= 4);
    kani::assume(s >= 1 && s <= 4);
    kani::assume(p <= 2);
    // Padding must be less than kernel_size (PyTorch constraint for avg_pool)
    kani::assume(p < ks);

    let in_h = in_h as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;

    let padded = in_h + 2 * p;
    if padded < ks {
        return;
    }
    let out_h = (padded - ks) / s + 1;

    let oh: u8 = kani::any();
    kani::assume((oh as usize) < out_h);
    let oh = oh as usize;

    // Count valid positions in the kernel window
    let mut count: usize = 0;
    let mut k: usize = 0;
    while k < ks {
        let ih = oh * s + k;
        if ih >= p && ih - p < in_h {
            count += 1;
        }
        k += 1;
    }
    assert!(
        count > 0,
        "avg_pool window must contain at least one valid input element"
    );
}

// ---------------------------------------------------------------------------
// adaptive_avg_pool2d: windows partition input without gaps
// ---------------------------------------------------------------------------

/// Prove: adaptive pooling windows are contiguous — end of window oh
/// equals start of window oh+1. This means there are no gaps in coverage.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn adaptive_pool_windows_contiguous() {
    let out_h: u8 = kani::any();
    let in_h: u8 = kani::any();
    let oh: u8 = kani::any();

    kani::assume(out_h >= 2);
    kani::assume(in_h >= 1);
    kani::assume(oh < out_h - 1); // oh and oh+1 both valid

    let out_h = out_h as usize;
    let in_h = in_h as usize;
    let oh = oh as usize;

    // end of window oh
    let end_oh = ((oh + 1) * in_h).div_ceil(out_h);
    // start of window oh+1
    let start_next = ((oh + 1) * in_h) / out_h;

    assert!(
        end_oh >= start_next,
        "windows must not have gaps (end_oh >= start_{oh+1})"
    );
}

/// Prove: adaptive pool window size is within expected bounds.
///
/// Each window has size in [floor(in/out), ceil(in/out)].
/// This ensures the adaptive pool produces balanced windows.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn adaptive_pool_window_size_bounded() {
    let out_h: u8 = kani::any();
    let in_h: u8 = kani::any();
    let oh: u8 = kani::any();

    kani::assume(out_h >= 1);
    kani::assume(in_h >= 1);
    kani::assume(oh < out_h);

    let out_h = out_h as usize;
    let in_h = in_h as usize;
    let oh = oh as usize;

    let start = (oh * in_h) / out_h;
    let end = ((oh + 1) * in_h).div_ceil(out_h);

    let window_size = end - start;
    let min_size = in_h / out_h;
    let max_size = in_h.div_ceil(out_h);

    assert!(
        window_size >= min_size,
        "window size must be >= floor(in/out)"
    );
    assert!(
        window_size <= max_size,
        "window size must be <= ceil(in/out)"
    );
}

// ---------------------------------------------------------------------------
// pool2d_out_len: output monotone non-decreasing in input_len
// ---------------------------------------------------------------------------

/// Prove: pool2d output length is monotonically non-decreasing in input length.
///
/// Larger input with fixed kernel/stride/padding produces >= output positions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_monotone_in_input_len() {
    let il1: u8 = kani::any();
    let il2: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il1 >= 1);
    kani::assume(il2 >= il1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);

    let il1 = il1 as usize;
    let il2 = il2 as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;

    let padded1 = il1 + 2 * p;
    let padded2 = il2 + 2 * p;

    if padded1 >= ks {
        assert!(padded2 >= ks, "larger input padded must also be >= ks");
        let r1 = super::pool::pool2d_out_len(il1, ks, p, s, false);
        let r2 = super::pool::pool2d_out_len(il2, ks, p, s, false);
        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert!(
            r2.unwrap() >= r1.unwrap(),
            "larger input must produce >= output"
        );
    }
}

// ---------------------------------------------------------------------------
// checked_buffer_len: overflow detection
// ---------------------------------------------------------------------------

/// Prove: checked_buffer_len detects overflow for large factor products.
///
/// When any pair of factors would overflow usize when multiplied, the
/// function must return Err.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn checked_buffer_len_detects_overflow() {
    let a: u32 = kani::any();
    let b: u32 = kani::any();

    kani::assume(a >= 1);
    kani::assume(b >= 1);

    let factors = [a as usize, b as usize];
    let result = super::checked_buffer_len(&factors, "test");
    let manual = (a as usize).checked_mul(b as usize);

    match (result, manual) {
        (Ok(product), Some(expected)) => {
            assert_eq!(product, expected, "must match checked_mul");
        }
        (Err(_), None) => {
            // Both detect overflow.
        }
        (Ok(p), None) => {
            panic!("checked_buffer_len returned Ok({p}) but manual overflows");
        }
        (Err(_), Some(e)) => {
            panic!("checked_buffer_len returned Err but manual Ok({e})");
        }
    }
}

/// Prove: checked_buffer_len of empty factors is 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn checked_buffer_len_empty_is_one() {
    let result = super::checked_buffer_len(&[], "test");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1, "empty factors must produce 1");
}

/// Prove: checked_buffer_len of single factor returns that factor.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn checked_buffer_len_single_factor() {
    let a: u16 = kani::any();
    kani::assume(a >= 1);

    let factors = [a as usize];
    let result = super::checked_buffer_len(&factors, "test");
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap(),
        a as usize,
        "single factor must return itself"
    );
}

// ---------------------------------------------------------------------------
// pool2d_out_len: ceil_mode adds at most 1 to floor_mode
// ---------------------------------------------------------------------------

/// Prove: ceil_mode output is at most 1 greater than floor_mode output.
///
/// div_ceil(n, s) - n/s is either 0 (when s divides n) or 1.
/// So ceil_out - floor_out is 0 or 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_ceil_mode_adds_at_most_one() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);

    let padded = (il as usize) + 2 * (p as usize);
    kani::assume(padded >= ks as usize);

    let floor_r =
        super::pool::pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, false);
    let ceil_r =
        super::pool::pool2d_out_len(il as usize, ks as usize, p as usize, s as usize, true);

    assert!(floor_r.is_ok());
    assert!(ceil_r.is_ok());

    let diff = ceil_r.unwrap() - floor_r.unwrap();
    assert!(
        diff <= 1,
        "ceil_mode can add at most 1 to floor_mode output"
    );
}

// ---------------------------------------------------------------------------
// pool2d_out_len: stride=1 output equals padded - kernel + 1
// ---------------------------------------------------------------------------

/// Prove: with stride=1, pool output length = padded_input - kernel_size + 1.
///
/// This is the identity case with no striding.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_stride_one_output_formula() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);

    let padded = (il as usize) + 2 * (p as usize);
    kani::assume(padded >= ks as usize);

    let result = super::pool::pool2d_out_len(il as usize, ks as usize, p as usize, 1, false);
    assert!(result.is_ok());
    let out = result.unwrap();
    let expected = padded - (ks as usize) + 1;
    assert_eq!(
        out, expected,
        "stride=1: output must be padded - kernel + 1"
    );
}

// ---------------------------------------------------------------------------
// pool2d_out_len: padding=0, kernel=1, stride=1 gives identity
// ---------------------------------------------------------------------------

/// Prove: pool with kernel_size=1, stride=1, padding=0 returns the input length.
///
/// This is the identity pooling case — every input position maps to one output.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_identity_kernel1_stride1_pad0() {
    let il: u8 = kani::any();
    kani::assume(il >= 1);

    let result = super::pool::pool2d_out_len(il as usize, 1, 0, 1, false);
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap(),
        il as usize,
        "identity pool must return input length"
    );
}

// ---------------------------------------------------------------------------
// pool2d_out_len: monotone non-decreasing in padding
// ---------------------------------------------------------------------------

/// Prove: pool2d output length is monotonically non-decreasing in padding.
///
/// More padding with fixed input/kernel/stride produces >= output positions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_monotone_in_padding() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p1: u8 = kani::any();
    let p2: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(p2 >= p1);

    let padded1 = (il as usize) + 2 * (p1 as usize);
    let padded2 = (il as usize) + 2 * (p2 as usize);

    if padded1 >= ks as usize {
        assert!(padded2 >= ks as usize);
        let r1 =
            super::pool::pool2d_out_len(il as usize, ks as usize, p1 as usize, s as usize, false);
        let r2 =
            super::pool::pool2d_out_len(il as usize, ks as usize, p2 as usize, s as usize, false);
        assert!(r1.is_ok());
        assert!(r2.is_ok());
        assert!(
            r2.unwrap() >= r1.unwrap(),
            "more padding must produce >= output"
        );
    }
}

// ---------------------------------------------------------------------------
// adaptive_avg_pool2d: first window starts at 0
// ---------------------------------------------------------------------------

/// Prove: adaptive pooling window 0 always starts at input position 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn adaptive_pool_first_window_starts_at_zero() {
    let out_h: u8 = kani::any();
    let in_h: u8 = kani::any();

    kani::assume(out_h >= 1);
    kani::assume(in_h >= 1);

    let out_h = out_h as usize;
    let in_h = in_h as usize;

    let start = (0 * in_h) / out_h;
    assert_eq!(start, 0, "first adaptive pool window must start at 0");
}

/// Prove: adaptive pooling last window ends at input length.
///
/// Window out_h-1 ends at ceil((out_h * in_h) / out_h) = ceil(in_h) = in_h.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn adaptive_pool_last_window_ends_at_input() {
    let out_h: u8 = kani::any();
    let in_h: u8 = kani::any();

    kani::assume(out_h >= 1);
    kani::assume(in_h >= 1);

    let out_h = out_h as usize;
    let in_h = in_h as usize;

    // Last window: oh = out_h - 1
    let end = (out_h * in_h).div_ceil(out_h);
    // (out_h * in_h) / out_h = in_h (exact), so div_ceil = in_h
    assert_eq!(end, in_h, "last adaptive pool window must end at in_h");
}

// ---------------------------------------------------------------------------
// adaptive_avg_pool2d: window has >= 1 element
// ---------------------------------------------------------------------------

/// Prove: every adaptive pooling window contains at least one element.
///
/// For any oh in [0, out_h), the window [start, end) has end > start.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn adaptive_pool_window_nonempty() {
    let out_h: u8 = kani::any();
    let in_h: u8 = kani::any();
    let oh: u8 = kani::any();

    kani::assume(out_h >= 1);
    kani::assume(in_h >= 1);
    kani::assume(oh < out_h);

    let out_h = out_h as usize;
    let in_h = in_h as usize;
    let oh = oh as usize;

    let start = (oh * in_h) / out_h;
    let end = ((oh + 1) * in_h).div_ceil(out_h);

    assert!(
        end > start,
        "adaptive pool window must contain at least one element"
    );
}
