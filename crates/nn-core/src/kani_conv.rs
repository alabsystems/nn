// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for DynTensor convolution output-length arithmetic
//! and narrow() bounds-check safety.
//!
//! These verify that the arithmetic in `conv1d_out_len`, `conv_transpose1d_out_len`,
//! and `narrow()` bounds checks cannot produce incorrect results through
//! integer overflow.
//!
//! The harnesses inline the arithmetic expressions from `dyn_tensor_conv.rs`
//! and `dyn_tensor_shape.rs` rather than importing the private functions.
//! This proves the arithmetic properties independent of error-handling paths.

#![cfg(kani)]

// ---------------------------------------------------------------------------
// conv1d_out_len arithmetic harnesses
// ---------------------------------------------------------------------------

/// Prove: conv1d output length arithmetic does not panic for small valid params.
///
/// Inlines: `effective_k = (ks - 1) * d + 1; padded = il + 2 * p;
/// out = (padded - effective_k) / s + 1`
///
/// Uses u8 inputs to cover the full small parameter space exhaustively.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_out_len_no_panic_small() {
    let il: u8 = kani::any(); // input_len
    let ks: u8 = kani::any(); // kernel_size
    let p: u8 = kani::any(); // padding
    let s: u8 = kani::any(); // stride
    let d: u8 = kani::any(); // dilation

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;
    let d = d as usize;

    // These operations match dyn_tensor_conv.rs:22-29
    let effective_k = (ks - 1) * d + 1;
    let padded = il + 2 * p;
    if padded >= effective_k {
        let out = (padded - effective_k) / s + 1;
        assert!(out >= 1, "conv1d output length must be >= 1 when valid");
    }
}

/// Prove: identity conv (k=1, s=1, d=1, p=0) preserves input length.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_out_len_identity() {
    let il: u8 = kani::any();
    kani::assume(il >= 1);

    let effective_k = (1 - 1) * 1 + 1; // = 1
    let padded = il as usize + 0; // = il
    let out = (padded - effective_k) / 1 + 1; // = (il - 1) + 1 = il
    assert_eq!(out, il as usize);
}

/// Prove: conv1d output length is monotonically non-decreasing in padding.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_out_len_monotone_in_padding() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p1: u8 = kani::any();
    let p2: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(il >= 1);
    kani::assume(p2 >= p1);

    let il = il as usize;
    let ks = ks as usize;
    let s = s as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;
    let padded1 = il + 2 * (p1 as usize);
    let padded2 = il + 2 * (p2 as usize);

    if padded1 >= effective_k {
        // p1 valid implies p2 valid (p2 >= p1 => padded2 >= padded1)
        assert!(padded2 >= effective_k, "more padding must also be valid");
        let o1 = (padded1 - effective_k) / s + 1;
        let o2 = (padded2 - effective_k) / s + 1;
        assert!(o2 >= o1, "more padding must produce >= output");
    }
}

// ---------------------------------------------------------------------------
// conv2d_out_len arithmetic harnesses
// ---------------------------------------------------------------------------

/// Prove: conv2d output length arithmetic does not panic for small valid params.
///
/// Inlines the same formula as conv1d_out_len (used per spatial dimension):
/// `effective_k = (ks - 1) * d + 1; padded = il + 2 * p;
/// out = (padded - effective_k) / s + 1`
///
/// Uses u8 inputs to cover the full small parameter space exhaustively.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv2d_out_len_no_panic_small() {
    let il: u8 = kani::any(); // input spatial dim (height or width)
    let ks: u8 = kani::any(); // kernel_size
    let p: u8 = kani::any(); // padding
    let s: u8 = kani::any(); // stride
    let d: u8 = kani::any(); // dilation

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;
    let d = d as usize;

    // Match dyn_tensor/conv/mod.rs:53-60 (conv2d_out_len)
    let effective_k = (ks - 1) * d + 1;
    let padded = il + 2 * p;
    if padded >= effective_k {
        let out = (padded - effective_k) / s + 1;
        assert!(out >= 1, "conv2d output length must be >= 1 when valid");
    }
}

/// Prove: conv2d identity conv (k=1, s=1, d=1, p=0) preserves spatial dimension.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv2d_out_len_identity() {
    let il: u8 = kani::any();
    kani::assume(il >= 1);

    let effective_k = (1 - 1) * 1 + 1; // = 1
    let padded = il as usize + 0; // = il
    let out = (padded - effective_k) / 1 + 1; // = il
    assert_eq!(out, il as usize);
}

/// Prove: conv2d output length is monotonically non-decreasing in padding.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv2d_out_len_monotone_in_padding() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p1: u8 = kani::any();
    let p2: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(il >= 1);
    kani::assume(p2 >= p1);

    let il = il as usize;
    let ks = ks as usize;
    let s = s as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;
    let padded1 = il + 2 * (p1 as usize);
    let padded2 = il + 2 * (p2 as usize);

    if padded1 >= effective_k {
        assert!(padded2 >= effective_k, "more padding must also be valid");
        let o1 = (padded1 - effective_k) / s + 1;
        let o2 = (padded2 - effective_k) / s + 1;
        assert!(o2 >= o1, "more padding must produce >= output");
    }
}

// ---------------------------------------------------------------------------
// conv_transpose1d_out_len arithmetic harnesses
// ---------------------------------------------------------------------------

/// Prove: conv_transpose1d output length arithmetic does not panic for small params.
///
/// Inlines: `positive = (il - 1) * s + d * (ks - 1) + op + 1; negative = 2 * p;
/// out = positive - negative`
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose1d_out_len_no_panic_small() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let op: u8 = kani::any(); // output_padding
    let s: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(il >= 1);
    kani::assume(op < s); // PyTorch constraint

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let op = op as usize;
    let s = s as usize;
    let d = d as usize;

    // Match dyn_tensor_conv.rs:44-52
    let positive = (il - 1) * s + d * (ks - 1) + op + 1;
    let negative = 2 * p;
    if negative <= positive {
        let out = positive - negative;
        assert!(out >= 1, "conv_transpose1d output must be >= 1 when valid");
    }
}

/// Prove: conv_transpose1d inverts conv1d for stride=1, dilation=1, output_padding=0.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose1d_inverts_conv1d_stride1() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume((il as usize) + 2 * (p as usize) >= ks as usize);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;

    // Forward conv1d: out = (il + 2*p - ks) / 1 + 1 = il + 2*p - ks + 1
    let effective_k = ks; // (ks-1)*1+1 = ks
    let padded = il + 2 * p;
    let mid = (padded - effective_k) / 1 + 1;

    // Inverse conv_transpose1d: out = (mid - 1)*1 + 1*(ks-1) + 0 + 1 - 2*p
    let positive = (mid - 1) + (ks - 1) + 1; // = mid + ks - 2 + 1 = mid + ks - 1
    let negative = 2 * p;
    if negative <= positive {
        let recovered = positive - negative;
        assert_eq!(
            recovered, il,
            "conv_transpose1d must recover original length for s=1, d=1, op=0"
        );
    }
}

// ---------------------------------------------------------------------------
// narrow() bounds-check overflow harness
// ---------------------------------------------------------------------------

/// Prove: when start + len overflows usize, the checked_add version
/// correctly rejects, even though the naive `start + len > dim_size`
/// may falsely accept (because the wrapped sum could be small).
///
/// This is a pure arithmetic proof of the overflow property in
/// dyn_tensor_shape.rs:137.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn narrow_bounds_check_overflow_safety() {
    let start: usize = kani::any();
    let len: usize = kani::any();
    let dim_size: usize = kani::any();
    kani::assume(dim_size >= 1);

    // Safe check using checked_add
    let safe_oob = start.checked_add(len).map_or(true, |sum| sum > dim_size);

    if !safe_oob {
        // If safe check says in-bounds, verify with u128 arithmetic
        let true_sum = start as u128 + len as u128;
        assert!(true_sum <= dim_size as u128);
    }
}

/// Prove: for small dimensions (u16), naive and safe checks agree.
///
/// This shows the overflow is only possible for pathological sizes
/// (start + len > usize::MAX), not for realistic tensor dimensions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn narrow_bounds_check_correct_for_small_dims() {
    let start: u16 = kani::any();
    let len: u16 = kani::any();
    let dim_size: u16 = kani::any();
    kani::assume(dim_size >= 1);

    let s = start as usize;
    let l = len as usize;
    let d = dim_size as usize;

    // For u16 values, s + l <= 131070, which fits in usize — no overflow.
    let naive_oob = s + l > d;
    let safe_oob = s.checked_add(l).map_or(true, |sum| sum > d);
    assert_eq!(naive_oob, safe_oob);
}
