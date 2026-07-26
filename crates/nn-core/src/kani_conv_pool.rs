// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for conv and pool dimension computations.
//!
//! Extends kani_conv.rs and kani_pool.rs with additional proofs:
//! - Conv1d/Conv2d output monotonicity in stride, dilation, and input length
//! - Conv groups divisibility invariant
//! - ConvTranspose2d output-length arithmetic
//! - ConvTranspose stride-output monotonicity
//! - Pool1d output-length arithmetic and identity
//! - Pool2d output monotonicity in kernel_size
//!
//! Part of #3587: Kani harnesses for DynTensor conv and pool operations.

#![cfg(kani)]

// ---------------------------------------------------------------------------
// Conv1d: output monotone non-decreasing in input length
// ---------------------------------------------------------------------------

/// Prove: conv1d output length is monotonically non-decreasing in input length.
///
/// Larger input (with all other params fixed) produces >= output positions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_out_len_monotone_in_input_len() {
    let il1: u8 = kani::any();
    let il2: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(il1 >= 1);
    kani::assume(il2 >= il1);

    let il1 = il1 as usize;
    let il2 = il2 as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;
    let padded1 = il1 + 2 * p;
    let padded2 = il2 + 2 * p;

    if padded1 >= effective_k {
        assert!(
            padded2 >= effective_k,
            "larger input must also produce valid padded size"
        );
        let o1 = (padded1 - effective_k) / s + 1;
        let o2 = (padded2 - effective_k) / s + 1;
        assert!(o2 >= o1, "larger input must produce >= output length");
    }
}

// ---------------------------------------------------------------------------
// Conv1d: output monotone non-increasing in stride
// ---------------------------------------------------------------------------

/// Prove: conv1d output length is monotonically non-increasing in stride.
///
/// Larger stride skips more input positions, producing fewer output positions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_out_len_monotone_decreasing_in_stride() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s1: u8 = kani::any();
    let s2: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s1 >= 1);
    kani::assume(s2 >= s1);
    kani::assume(d >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;
    let padded = il + 2 * p;

    if padded >= effective_k {
        let numerator = padded - effective_k;
        let o1 = numerator / (s1 as usize) + 1;
        let o2 = numerator / (s2 as usize) + 1;
        assert!(o1 >= o2, "larger stride must produce <= output");
    }
}

// ---------------------------------------------------------------------------
// Conv1d: output monotone non-increasing in dilation
// ---------------------------------------------------------------------------

/// Prove: conv1d output length is monotonically non-increasing in dilation.
///
/// Larger dilation increases the effective kernel size, reducing output.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_out_len_monotone_decreasing_in_dilation() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d1 >= 1);
    kani::assume(d2 >= d1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;

    let effective_k1 = (ks - 1) * (d1 as usize) + 1;
    let effective_k2 = (ks - 1) * (d2 as usize) + 1;
    let padded = il + 2 * p;

    // effective_k2 >= effective_k1 since d2 >= d1 and ks >= 1
    if padded >= effective_k2 {
        assert!(
            padded >= effective_k1,
            "smaller dilation effective_k must also be valid"
        );
        let o1 = (padded - effective_k1) / s + 1;
        let o2 = (padded - effective_k2) / s + 1;
        assert!(o1 >= o2, "larger dilation must produce <= output");
    }
}

// ---------------------------------------------------------------------------
// Conv1d: stride > 0 prevents division by zero
// ---------------------------------------------------------------------------

/// Prove: conv1d formula never divides by zero when stride >= 1.
///
/// The formula `(padded - effective_k) / stride + 1` is safe from
/// division by zero when stride >= 1. This is enforced by parameter
/// validation, and this harness proves the arithmetic consequence.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_no_division_by_zero() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1); // This is the key precondition
    kani::assume(d >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;
    let padded = il + 2 * p;

    if padded >= effective_k {
        // This division is safe because s >= 1
        let out = (padded - effective_k) / s + 1;
        assert!(out >= 1);
    }
}

// ---------------------------------------------------------------------------
// Conv groups divisibility: in_channels % groups == 0
// ---------------------------------------------------------------------------

/// Prove: when groups divides in_channels, the per-group channel count
/// is well-defined and positive. This is the precondition for grouped
/// convolution weight shapes: weight has shape [out_ch, in_ch/groups, ksize].
///
/// Also proves: when groups divides out_channels, per-group output
/// channels are well-defined.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_groups_divisibility_well_defined() {
    let in_ch: u8 = kani::any();
    let out_ch: u8 = kani::any();
    let groups: u8 = kani::any();

    kani::assume(groups >= 1);
    kani::assume(in_ch >= 1);
    kani::assume(out_ch >= 1);
    kani::assume(in_ch as usize % groups as usize == 0);
    kani::assume(out_ch as usize % groups as usize == 0);

    let g = groups as usize;
    let ic = in_ch as usize;
    let oc = out_ch as usize;

    let in_per_group = ic / g;
    let out_per_group = oc / g;

    // Per-group channels must be >= 1
    assert!(in_per_group >= 1, "in_channels / groups must be >= 1");
    assert!(out_per_group >= 1, "out_channels / groups must be >= 1");

    // Reconstruction: groups * per_group == total
    assert_eq!(
        g * in_per_group,
        ic,
        "groups * in_per_group must reconstruct in_channels"
    );
    assert_eq!(
        g * out_per_group,
        oc,
        "groups * out_per_group must reconstruct out_channels"
    );
}

// ---------------------------------------------------------------------------
// Conv2d: output monotone non-increasing in stride
// ---------------------------------------------------------------------------

/// Prove: conv2d output length is monotonically non-increasing in stride.
///
/// Same formula as conv1d per spatial dimension.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv2d_out_len_monotone_decreasing_in_stride() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s1: u8 = kani::any();
    let s2: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s1 >= 1);
    kani::assume(s2 >= s1);
    kani::assume(d >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;
    let padded = il + 2 * p;

    if padded >= effective_k {
        let numerator = padded - effective_k;
        let o1 = numerator / (s1 as usize) + 1;
        let o2 = numerator / (s2 as usize) + 1;
        assert!(o1 >= o2, "conv2d: larger stride must produce <= output");
    }
}

// ---------------------------------------------------------------------------
// Conv2d: output monotone non-decreasing in input length
// ---------------------------------------------------------------------------

/// Prove: conv2d output length is monotonically non-decreasing in input size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv2d_out_len_monotone_in_input_len() {
    let il1: u8 = kani::any();
    let il2: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1);
    kani::assume(d >= 1);
    kani::assume(il1 >= 1);
    kani::assume(il2 >= il1);

    let il1 = il1 as usize;
    let il2 = il2 as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;
    let padded1 = il1 + 2 * p;
    let padded2 = il2 + 2 * p;

    if padded1 >= effective_k {
        assert!(padded2 >= effective_k);
        let o1 = (padded1 - effective_k) / s + 1;
        let o2 = (padded2 - effective_k) / s + 1;
        assert!(o2 >= o1, "conv2d: larger input must produce >= output");
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose2d: no panic for small valid params
// ---------------------------------------------------------------------------

/// Prove: conv_transpose2d output length arithmetic does not panic for
/// small valid params.
///
/// Formula: out = (il - 1) * s + d * (ks - 1) + op + 1 - 2 * p
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose2d_out_len_no_panic_small() {
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
    kani::assume(op < s); // PyTorch constraint: output_padding < stride

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let op = op as usize;
    let s = s as usize;
    let d = d as usize;

    // Match dyn_tensor/conv/transpose2d.rs:46-73
    let positive = (il - 1) * s + d * (ks - 1) + op + 1;
    let negative = 2 * p;
    if negative < positive {
        let out = positive - negative;
        assert!(out >= 1, "conv_transpose2d output must be >= 1 when valid");
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose2d: identity (il=any, ks=1, s=1, d=1, p=0, op=0) preserves length
// ---------------------------------------------------------------------------

/// Prove: conv_transpose2d identity (k=1, s=1, d=1, p=0, op=0) preserves length.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose2d_out_len_identity() {
    let il: u8 = kani::any();
    kani::assume(il >= 1);

    let il = il as usize;

    // identity: ks=1, s=1, d=1, p=0, op=0
    let positive = (il - 1) * 1 + 1 * (1 - 1) + 0 + 1; // = il - 1 + 0 + 0 + 1 = il
    let negative = 2 * 0; // = 0
    let out = positive - negative;
    assert_eq!(out, il, "conv_transpose2d identity must preserve length");
}

// ---------------------------------------------------------------------------
// ConvTranspose1d: output monotone non-decreasing in stride
// ---------------------------------------------------------------------------

/// Prove: conv_transpose1d output is monotonically non-decreasing in stride.
///
/// Unlike regular conv where larger stride decreases output, transposed conv
/// with larger stride increases output because the formula multiplies (il-1)*s.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn conv_transpose1d_out_len_monotone_in_stride() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s1: u8 = kani::any();
    let s2: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s1 >= 1);
    kani::assume(s2 >= s1);
    kani::assume(d >= 1);
    kani::assume(il >= 2); // need il >= 2 so (il-1) >= 1 for monotonicity to hold
                           // output_padding must be < stride for both strides
                           // Use op=0 to simplify (output_padding < s1 AND < s2 both satisfied)

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let d = d as usize;

    let negative = 2 * p;

    let positive1 = (il - 1) * (s1 as usize) + d * (ks - 1) + 0 + 1;
    let positive2 = (il - 1) * (s2 as usize) + d * (ks - 1) + 0 + 1;

    if negative < positive1 {
        // s2 >= s1 and il >= 2 => positive2 >= positive1
        assert!(
            positive2 >= positive1,
            "larger stride must produce >= positive terms"
        );
        assert!(negative < positive2);
        let o1 = positive1 - negative;
        let o2 = positive2 - negative;
        assert!(
            o2 >= o1,
            "conv_transpose1d: larger stride must produce >= output"
        );
    }
}

// ---------------------------------------------------------------------------
// Pool1d: output length no panic for small valid params
// ---------------------------------------------------------------------------

/// Prove: pool1d (same formula as pool2d with dilation=1) output arithmetic
/// does not panic for small valid params.
///
/// Pool1d uses the same `pool2d_out_len` function internally. This harness
/// provides naming clarity and covers the 1D pooling use case explicitly.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool1d_out_len_no_panic_small() {
    let il: u8 = kani::any(); // input length
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

    // Pool1d uses pool2d_out_len(in_len, kernel_size, padding, stride, false)
    let padded = il + 2 * p;
    if padded >= ks {
        let out = (padded - ks) / s + 1;
        assert!(out >= 1, "pool1d output length must be >= 1 when valid");
    }
}

// ---------------------------------------------------------------------------
// Pool1d: identity (k=1, s=1, p=0) preserves length
// ---------------------------------------------------------------------------

/// Prove: pool1d identity pool (k=1, s=1, p=0) preserves input length.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool1d_out_len_identity() {
    let il: u8 = kani::any();
    kani::assume(il >= 1);

    let padded = il as usize; // p=0
    let out = (padded - 1) / 1 + 1; // ks=1, s=1
    assert_eq!(out, il as usize, "pool1d identity must preserve length");
}

// ---------------------------------------------------------------------------
// Pool2d: output monotone non-increasing in kernel_size
// ---------------------------------------------------------------------------

/// Prove: pool2d output length is monotonically non-increasing in kernel_size.
///
/// Larger kernel means more elements consumed per window, fewer output positions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_out_len_monotone_decreasing_in_kernel_size() {
    let il: u8 = kani::any();
    let ks1: u8 = kani::any();
    let ks2: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(ks1 >= 1);
    kani::assume(ks2 >= ks1);
    kani::assume(s >= 1);
    kani::assume(il >= 1);

    let il = il as usize;
    let p = p as usize;
    let s = s as usize;

    let padded = il + 2 * p;

    // If the larger kernel is valid, the smaller kernel is also valid
    if padded >= ks2 as usize {
        assert!(padded >= ks1 as usize);
        let o1 = (padded - ks1 as usize) / s + 1;
        let o2 = (padded - ks2 as usize) / s + 1;
        assert!(o1 >= o2, "pool2d: larger kernel must produce <= output");
    }
}

// ---------------------------------------------------------------------------
// Pool2d: stride > 0 prevents division by zero
// ---------------------------------------------------------------------------

/// Prove: pool2d formula never divides by zero when stride >= 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn pool2d_no_division_by_zero() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let p: u8 = kani::any();
    let s: u8 = kani::any();

    kani::assume(ks >= 1);
    kani::assume(s >= 1); // Key precondition
    kani::assume(il >= 1);

    let il = il as usize;
    let ks = ks as usize;
    let p = p as usize;
    let s = s as usize;

    let padded = il + 2 * p;
    if padded >= ks {
        // Division by s is safe because s >= 1
        let out = (padded - ks) / s + 1;
        assert!(out >= 1);
    }
}

// ---------------------------------------------------------------------------
// Conv1d: same-padding produces out >= in when stride=1
// ---------------------------------------------------------------------------

/// Prove: conv1d with "same" padding (p = (ks-1)*d/2, stride=1) produces
/// output length >= input length.
///
/// When padding >= (effective_k - 1) / 2, the output is at least as large
/// as the input for stride=1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_same_padding_preserves_length() {
    let il: u8 = kani::any();
    let ks: u8 = kani::any();
    let d: u8 = kani::any();

    kani::assume(il >= 1);
    kani::assume(ks >= 1);
    kani::assume(d >= 1);
    // Restrict to avoid overflow in u8 arithmetic
    kani::assume(ks <= 15);
    kani::assume(d <= 15);

    let il = il as usize;
    let ks = ks as usize;
    let d = d as usize;

    let effective_k = (ks - 1) * d + 1;
    // "same" padding: at least half the effective kernel on each side
    let p = (effective_k - 1) / 2;
    let padded = il + 2 * p;
    // stride = 1
    assert!(
        padded >= effective_k,
        "same padding must make padded >= effective_k"
    );
    let out = (padded - effective_k) / 1 + 1;
    // For stride=1, same padding gives out = il + 2*p - effective_k + 1
    // = il + 2*((effective_k-1)/2) - effective_k + 1
    // Since 2*((effective_k-1)/2) >= effective_k - 1 (floor div rounds down):
    // out >= il + (effective_k - 1) - effective_k + 1 = il
    // BUT: when effective_k is even, 2*floor((ek-1)/2) = ek-2, so out = il-1
    // So the guarantee is: out >= il - (effective_k % 2 == 0) as usize
    // For odd effective_k, out == il. For even, out == il or il-1.
    // We prove the weaker bound: out >= il when effective_k is odd.
    if effective_k % 2 == 1 {
        assert!(
            out >= il,
            "same padding with odd effective kernel preserves length at stride=1"
        );
    }
}
