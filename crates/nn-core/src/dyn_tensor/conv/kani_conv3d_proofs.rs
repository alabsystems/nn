// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Conv3d safety properties (#3867).
//!
//! Proves correctness of `conv3d_out_len` — the spatial dimension formula used
//! by `DynTensor::conv3d` for 3D patch embeddings (Qwen3-VL vision encoder):
//!
//! 1. No panic for valid inputs (kernel_size > 0, stride > 0, dilation > 0)
//! 2. Output length is positive for valid configurations
//! 3. Same-padding with odd kernel and stride=1 preserves dimension
//! 4. Stride-2 with k=3, p=1 halves even dimensions
//! 5. Zero kernel_size is rejected
//! 6. Zero stride is rejected
//! 7. Zero dilation is rejected
//! 8. Undersized padded input is rejected
//! 9. Conv3d output formula is consistent across all 3 spatial dimensions

#![cfg(kani)]

use super::conv3d::conv3d_out_len;

// ---------------------------------------------------------------------------
// Harness 1: conv3d_out_len does not panic for valid inputs
// ---------------------------------------------------------------------------

/// Prove: `conv3d_out_len` returns `Ok` (does not panic) for any valid parameter
/// combination where kernel_size, stride, and dilation are positive and the
/// effective kernel fits within the padded input.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv3d_out_len_no_panic() {
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();
    let stride: usize = kani::any();
    let padding: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(input_len > 0 && input_len <= 256);
    kani::assume(kernel_size > 0 && kernel_size <= 16);
    kani::assume(stride > 0 && stride <= 8);
    kani::assume(padding <= 16);
    kani::assume(dilation > 0 && dilation <= 4);

    // Ensure effective kernel fits in padded input
    let effective_k = (kernel_size - 1) * dilation + 1;
    let padded = input_len + 2 * padding;
    kani::assume(padded >= effective_k);

    let result = conv3d_out_len(input_len, kernel_size, padding, stride, dilation);
    assert!(
        result.is_ok(),
        "conv3d_out_len must succeed for valid inputs"
    );
}

// ---------------------------------------------------------------------------
// Harness 2: Output length is positive for valid configurations
// ---------------------------------------------------------------------------

/// Prove: when `conv3d_out_len` succeeds, the output is always > 0.
///
/// The conv3d output formula is: `(padded - effective_k) / stride + 1`.
/// Since `padded >= effective_k` (validated) and stride >= 1, the minimum
/// output is 1 (when padded == effective_k).
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv3d_out_len_positive() {
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();
    let stride: usize = kani::any();
    let padding: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 128);
    kani::assume(kernel_size >= 1 && kernel_size <= 8);
    kani::assume(stride >= 1 && stride <= 4);
    kani::assume(padding <= 8);
    kani::assume(dilation >= 1 && dilation <= 2);

    let effective_k = (kernel_size - 1) * dilation + 1;
    let padded = input_len + 2 * padding;
    kani::assume(padded >= effective_k);

    let result = conv3d_out_len(input_len, kernel_size, padding, stride, dilation);
    assert!(result.is_ok(), "must succeed for valid inputs");
    let out = result.unwrap();
    assert!(out > 0, "output length must be positive for valid config");
}

// ---------------------------------------------------------------------------
// Harness 3: Same-padding preserves dimension (stride=1, odd kernel)
// ---------------------------------------------------------------------------

/// Prove: with stride=1, dilation=1, odd kernel, and padding=kernel/2,
/// `conv3d_out_len` returns the input dimension unchanged.
///
/// This is the "same-padding" property: for odd kernels with symmetric padding
/// and unit stride, the output spatial dimension equals the input.
/// Formula: out = (in + 2*(k/2) - k) / 1 + 1 = (in + k-1 - k) + 1 = in
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv3d_same_padding_preserves_dim() {
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 64);
    kani::assume(kernel_size >= 1 && kernel_size <= 7);
    kani::assume(kernel_size % 2 == 1); // odd kernel for symmetric padding

    let padding = kernel_size / 2;
    let stride = 1;
    let dilation = 1;

    let result = conv3d_out_len(input_len, kernel_size, padding, stride, dilation);
    assert!(result.is_ok(), "same-padding config must succeed");
    let out = result.unwrap();
    assert_eq!(out, input_len, "same padding should preserve dimension");
}

// ---------------------------------------------------------------------------
// Harness 4: Stride-2 with k=3, p=1 halves even dimensions
// ---------------------------------------------------------------------------

/// Prove: with kernel=3, padding=1, stride=2, dilation=1, and even input,
/// the output dimension is exactly input_len / 2.
///
/// Formula: out = (in + 2*1 - 3) / 2 + 1 = (in - 1) / 2 + 1 = in/2 (for even in)
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv3d_stride2_halves() {
    let input_len: usize = kani::any();
    kani::assume(input_len >= 4 && input_len <= 128);
    kani::assume(input_len % 2 == 0); // even for clean halving

    let kernel_size = 3;
    let padding = 1;
    let stride = 2;
    let dilation = 1;

    let result = conv3d_out_len(input_len, kernel_size, padding, stride, dilation);
    assert!(result.is_ok(), "stride-2 config must succeed");
    let out = result.unwrap();
    assert_eq!(out, input_len / 2, "stride 2 with k=3 p=1 should halve");
}

// ---------------------------------------------------------------------------
// Harness 5: Zero kernel_size is rejected
// ---------------------------------------------------------------------------

/// Prove: `conv3d_out_len` returns `Err` when kernel_size == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv3d_rejects_zero_kernel() {
    let input_len: usize = kani::any();
    let stride: usize = kani::any();
    let padding: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 64);
    kani::assume(stride >= 1 && stride <= 4);
    kani::assume(padding <= 8);
    kani::assume(dilation >= 1 && dilation <= 4);

    let result = conv3d_out_len(input_len, 0, padding, stride, dilation);
    assert!(result.is_err(), "kernel_size=0 must be rejected");
}

// ---------------------------------------------------------------------------
// Harness 6: Zero stride is rejected
// ---------------------------------------------------------------------------

/// Prove: `conv3d_out_len` returns `Err` when stride == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv3d_rejects_zero_stride() {
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();
    let padding: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 64);
    kani::assume(kernel_size >= 1 && kernel_size <= 8);
    kani::assume(padding <= 8);
    kani::assume(dilation >= 1 && dilation <= 4);

    let result = conv3d_out_len(input_len, kernel_size, padding, 0, dilation);
    assert!(result.is_err(), "stride=0 must be rejected");
}

// ---------------------------------------------------------------------------
// Harness 7: Zero dilation is rejected
// ---------------------------------------------------------------------------

/// Prove: `conv3d_out_len` returns `Err` when dilation == 0.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv3d_rejects_zero_dilation() {
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();
    let padding: usize = kani::any();
    let stride: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 64);
    kani::assume(kernel_size >= 1 && kernel_size <= 8);
    kani::assume(padding <= 8);
    kani::assume(stride >= 1 && stride <= 4);

    let result = conv3d_out_len(input_len, kernel_size, padding, stride, 0);
    assert!(result.is_err(), "dilation=0 must be rejected");
}

// ---------------------------------------------------------------------------
// Harness 8: Undersized padded input is rejected
// ---------------------------------------------------------------------------

/// Prove: `conv3d_out_len` returns `Err` when the padded input is smaller
/// than the effective kernel size.
///
/// This guards against the formula producing a nonsensical negative numerator.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv3d_rejects_undersized_padded_input() {
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 16);
    kani::assume(kernel_size >= 2 && kernel_size <= 8);
    kani::assume(dilation >= 1 && dilation <= 4);

    let effective_k = (kernel_size - 1) * dilation + 1;
    // No padding — the input alone is smaller than the effective kernel
    kani::assume(input_len < effective_k);

    let result = conv3d_out_len(input_len, kernel_size, 0, 1, dilation);
    assert!(
        result.is_err(),
        "padded input smaller than effective kernel must be rejected"
    );
}

// ---------------------------------------------------------------------------
// Harness 9: Output formula consistent across 3 spatial dimensions
// ---------------------------------------------------------------------------

/// Prove: `conv3d_out_len` produces the same result regardless of which
/// spatial dimension (D, H, W) it's computing, given identical parameters.
///
/// This verifies the function is stateless — it depends only on its arguments,
/// not on any hidden state that might differ between the three calls in
/// `DynTensor::conv3d`.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv3d_out_len_consistent_across_dims() {
    let input_len: usize = kani::any();
    let kernel_size: usize = kani::any();
    let padding: usize = kani::any();
    let stride: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(input_len >= 1 && input_len <= 64);
    kani::assume(kernel_size >= 1 && kernel_size <= 8);
    kani::assume(padding <= 4);
    kani::assume(stride >= 1 && stride <= 4);
    kani::assume(dilation >= 1 && dilation <= 2);

    let effective_k = (kernel_size - 1) * dilation + 1;
    let padded = input_len + 2 * padding;
    kani::assume(padded >= effective_k);

    // Call the function twice with identical parameters
    let r1 = conv3d_out_len(input_len, kernel_size, padding, stride, dilation);
    let r2 = conv3d_out_len(input_len, kernel_size, padding, stride, dilation);

    assert!(r1.is_ok() && r2.is_ok(), "both calls must succeed");
    assert_eq!(
        r1.unwrap(),
        r2.unwrap(),
        "identical inputs must produce identical outputs"
    );
}
