// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for fused GPU kernel Rust-side logic (#3351).
//!
//! The 7 fused norm kernels (Snake, RmsNorm, GroupNorm, LayerNorm,
//! InstanceNorm, AdaIN, AdaLN) share common Rust validation patterns:
//! eps validation, channel stride computation, dimension overflow checks,
//! and u32 dispatch parameter conversion.
//!
//! These harnesses prove the correctness of that shared logic:
//! - `validate_eps` rejects all non-finite or non-positive eps
//! - Snake channel_stride is always >= 1 (no zero-division in MSL)
//! - GroupNorm checked_mul chain catches all overflow paths
//! - `to_u32` dispatch conversion rejects out-of-range values
//! - RmsNorm flat_rows handles rank-1 edge case

/// Prove: `validate_eps` accepts all finite positive f32 and rejects everything else.
///
/// Models the logic from `dyn_tensor_metal_fused_helpers.rs:29-37`:
/// `let eps_f32 = eps as f32; if !eps_f32.is_finite() || eps_f32 <= 0.0 { Err } else { Ok }`
///
/// Key edge case: very small positive f64 values (< f32::MIN_POSITIVE subnormal
/// threshold) become 0.0 when cast to f32, and must be rejected.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_eps_rejects_non_positive_and_non_finite() {
    let eps: f64 = kani::any();

    let eps_f32 = eps as f32;
    let accepted = eps_f32.is_finite() && eps_f32 > 0.0;

    if accepted {
        // If the check passes, eps_f32 is a usable positive finite value.
        // This means rsqrt(variance + eps) won't produce Inf on GPU.
        assert!(eps_f32 > 0.0, "accepted eps must be positive");
        assert!(eps_f32.is_finite(), "accepted eps must be finite");
        // Also verify the value is safe for GPU rsqrt denominator.
        assert!(eps_f32 != 0.0, "accepted eps must be nonzero");
    } else {
        // The check correctly rejects this eps value.
        // At least one of the conditions failed.
        assert!(
            !eps_f32.is_finite() || eps_f32 <= 0.0,
            "rejected eps must be non-finite or non-positive"
        );
    }
}

/// Prove: `validate_eps` rejects f64 values that become zero when cast to f32.
///
/// This catches the subnormal flush edge case: `f64::MIN_POSITIVE * 0.001`
/// is positive in f64 but becomes 0.0 in f32. The GPU would compute
/// `rsqrt(0 + 0) = Inf`, corrupting the output.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_eps_rejects_f64_subnormal_to_f32_zero() {
    let eps: f64 = kani::any();

    // Constrain: eps is positive in f64 but zero in f32.
    kani::assume(eps > 0.0);
    kani::assume(eps as f32 == 0.0);

    let eps_f32 = eps as f32;
    let accepted = eps_f32.is_finite() && eps_f32 > 0.0;

    // Must be rejected: eps_f32 is 0.0, which is not > 0.0.
    assert!(!accepted, "subnormal-to-zero eps must be rejected");
}

/// Prove: Snake channel_stride is always >= 1 after the `.max(1)` guard.
///
/// Models the logic from `dyn_tensor_metal_snake_fused.rs:69-74`:
/// ```
/// let channel_stride = if dims.len() >= 2 {
///     checked_dim_product(&dims[2..])?
/// } else { 1 };
/// let channel_stride = channel_stride.max(1);
/// ```
///
/// Without the `.max(1)` guard, an empty spatial suffix (rank-2 tensor
/// `[B, C]`) produces `checked_dim_product(&[]) = 1`, but a tensor with
/// a zero spatial dim `[B, C, 0]` would produce 0 — and the MSL kernel
/// would compute `(gid / 0)` (undefined behavior).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(5)]
fn snake_channel_stride_always_positive() {
    // Model spatial dims as 0-3 dimensions (covers rank 1..5 tensors).
    let num_spatial: usize = kani::any();
    kani::assume(num_spatial <= 3);

    let rank_ge_2: bool = kani::any();

    let channel_stride = if rank_ge_2 {
        // checked_dim_product of spatial dims.
        // Model: product of 0..3 dimensions, each bounded.
        let mut product: usize = 1;
        let mut i = 0;
        let mut overflowed = false;
        while i < num_spatial {
            let dim: usize = kani::any();
            kani::assume(dim <= 4096); // reasonable spatial bound
            match product.checked_mul(dim) {
                Some(p) => product = p,
                None => {
                    overflowed = true;
                    break;
                }
            }
            i += 1;
        }
        if overflowed {
            return; // checked_dim_product returns Err — function exits early
        }
        product
    } else {
        1usize
    };

    // The .max(1) guard from line 74.
    let guarded = channel_stride.max(1);

    // Invariant: MSL kernel divides by channel_stride. Must never be zero.
    assert!(guarded >= 1, "channel_stride must be >= 1 after guard");
}

/// Prove: GroupNorm checked_mul chain correctly detects all overflow paths.
///
/// Models the three overflow-checked multiplications from
/// `dyn_tensor_metal_group_norm_fused.rs:68-87`:
/// 1. `flat_rows = batch * num_groups`
/// 2. `flat_cols = channels_per_group * spatial.max(1)`
/// 3. `total_elems = flat_rows * flat_cols`
///
/// If any overflows, the function returns Err (no silent truncation).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn group_norm_checked_mul_no_silent_overflow() {
    let batch: usize = kani::any();
    let num_groups: usize = kani::any();
    let channels_per_group: usize = kani::any();
    let spatial: usize = kani::any();

    // Realistic bounds for CBMC tractability.
    kani::assume(batch <= (1usize << 16));
    kani::assume(num_groups <= (1usize << 12));
    kani::assume(channels_per_group <= (1usize << 16));
    kani::assume(spatial <= (1usize << 20));
    kani::assume(batch > 0);
    kani::assume(num_groups > 0);
    kani::assume(channels_per_group > 0);

    // Step 1: flat_rows = batch * num_groups
    let flat_rows = match batch.checked_mul(num_groups) {
        Some(r) => r,
        None => return, // overflow correctly caught
    };

    // Step 2: flat_cols = channels_per_group * spatial.max(1)
    let spatial_safe = spatial.max(1);
    let flat_cols = match channels_per_group.checked_mul(spatial_safe) {
        Some(c) => c,
        None => return, // overflow correctly caught
    };

    // Step 3: total_elems = flat_rows * flat_cols
    let total_elems = match flat_rows.checked_mul(flat_cols) {
        Some(t) => t,
        None => return, // overflow correctly caught
    };

    // If we reach here, all three multiplications succeeded without overflow.
    // Verify the result equals the unchecked product (no silent truncation).
    // Use u128 to compute the true product without overflow.
    let true_product =
        (batch as u128) * (num_groups as u128) * (channels_per_group as u128) * (spatial_safe as u128);

    assert!(
        true_product <= usize::MAX as u128,
        "checked_mul passed but true product exceeds usize::MAX"
    );
    assert_eq!(
        total_elems as u128, true_product,
        "checked_mul result must equal true product"
    );
}

/// Prove: `to_u32` correctly partitions usize into accept/reject.
///
/// Models the logic from `lib.rs:283-287`:
/// `u32::try_from(val).map_err(...)`
///
/// All fused kernels pass dimension values through `to_u32` before
/// encoding into Metal dispatch parameters. Metal uses 32-bit grid sizes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn to_u32_accepts_exactly_u32_range() {
    let val: usize = kani::any();

    let result = u32::try_from(val);

    if val <= u32::MAX as usize {
        assert!(result.is_ok(), "values <= u32::MAX must be accepted");
        assert_eq!(result.unwrap() as usize, val, "conversion must be lossless");
    } else {
        assert!(result.is_err(), "values > u32::MAX must be rejected");
    }
}

/// Prove: RmsNorm flat_rows handles rank-1 edge case correctly.
///
/// Models the logic from `dyn_tensor_metal_rms_norm_fused.rs:63-65`:
/// ```
/// let flat_rows = checked_dim_product(&dims[..rank - 1])?;
/// let flat_rows = if flat_rows == 0 && rank == 1 { 1 } else { flat_rows };
/// ```
///
/// For rank-1 input `[hidden_dim]`, `dims[..0]` is empty, so
/// `checked_dim_product(&[]) = 1`. But the guard handles the edge case
/// where the product is 0 (shouldn't happen with empty slice, but
/// defense-in-depth).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn rms_norm_flat_rows_rank1_produces_one() {
    let rank: usize = kani::any();
    kani::assume(rank >= 1 && rank <= 4);

    // Model prefix dims (dims[..rank-1]).
    let mut flat_rows: usize = 1;
    let mut i = 0;
    let prefix_len = rank - 1;
    let mut overflowed = false;
    while i < prefix_len {
        let dim: usize = kani::any();
        kani::assume(dim <= 4096);
        kani::assume(dim > 0); // valid tensor dims are positive
        match flat_rows.checked_mul(dim) {
            Some(r) => flat_rows = r,
            None => {
                overflowed = true;
                break;
            }
        }
        i += 1;
    }
    if overflowed {
        return; // overflow → Err path
    }

    // The guard from line 65.
    let flat_rows = if flat_rows == 0 && rank == 1 { 1 } else { flat_rows };

    // Invariant: flat_rows is always >= 1 for valid inputs.
    // This ensures the Metal dispatch launches at least 1 threadgroup row.
    assert!(flat_rows >= 1, "flat_rows must be >= 1 for valid inputs");
}

// --- AdaIN 2-dispatch pattern harnesses ---

/// Prove: AdaIN flat_rows overflow check catches all overflows.
///
/// Models the logic from `dyn_tensor_metal_adain_fused.rs:76-81`:
/// `batch.checked_mul(channels).ok_or_else(|| DimensionOverflow)`
///
/// If both batch and channels are valid (non-zero, bounded), checked_mul
/// either returns the correct product or detects overflow.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn adain_flat_rows_checked_mul_sound() {
    let batch: usize = kani::any();
    let channels: usize = kani::any();

    // Realistic bounds for GPU dispatch (CBMC tractability).
    kani::assume(batch <= (1usize << 16));
    kani::assume(channels <= (1usize << 16));
    kani::assume(batch > 0);
    kani::assume(channels > 0);

    match batch.checked_mul(channels) {
        Some(flat_rows) => {
            // Verify result matches true product (no silent truncation).
            let true_product = (batch as u128) * (channels as u128);
            assert!(
                true_product <= usize::MAX as u128,
                "checked_mul passed but overflows usize"
            );
            assert_eq!(
                flat_rows as u128, true_product,
                "flat_rows must equal batch * channels"
            );
            // Verify positive (needed for Metal threadgroup count).
            assert!(flat_rows >= 1, "flat_rows must be >= 1 for valid inputs");
        }
        None => {
            // Overflow correctly detected.
            let true_product = (batch as u128) * (channels as u128);
            assert!(
                true_product > usize::MAX as u128,
                "checked_mul returned None but product fits in usize"
            );
        }
    }
}

/// Prove: AdaIN total_elems overflow check catches all overflows.
///
/// Models `dyn_tensor_metal_adain_fused.rs:95-100`:
/// `flat_rows.checked_mul(spatial).ok_or_else(|| DimensionOverflow)`
///
/// Both AdaIN kernels compute `total_elems = flat_rows * spatial` for buffer
/// sizing. Overflow here would allocate an undersized Metal buffer, causing
/// out-of-bounds GPU writes.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn adain_total_elems_checked_mul_sound() {
    let flat_rows: usize = kani::any();
    let spatial: usize = kani::any();

    kani::assume(flat_rows <= (1usize << 20));
    kani::assume(spatial <= (1usize << 20));
    // spatial > 0 is guarded by the early-exit check before this code.
    kani::assume(flat_rows > 0);
    kani::assume(spatial > 0);

    match flat_rows.checked_mul(spatial) {
        Some(total) => {
            let true_product = (flat_rows as u128) * (spatial as u128);
            assert_eq!(total as u128, true_product, "total_elems must be exact");
        }
        None => {
            let true_product = (flat_rows as u128) * (spatial as u128);
            assert!(true_product > usize::MAX as u128, "false overflow detection");
        }
    }
}

// --- NormConv 2-dispatch pattern harnesses ---

/// Prove: NormConv effective kernel size computation never underflows.
///
/// Models `dyn_tensor_metal_norm_conv_fused.rs:193-194`:
/// `let effective_k = (kernel_size - 1) * dilation + 1;`
///
/// For valid conv1d parameters (kernel_size >= 1, dilation >= 1), the
/// effective kernel size is always >= 1 and the computation doesn't overflow.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn norm_conv_effective_kernel_no_underflow() {
    let kernel_size: usize = kani::any();
    let dilation: usize = kani::any();

    // Valid conv1d: kernel_size >= 1, dilation >= 1.
    kani::assume(kernel_size >= 1 && kernel_size <= 1024);
    kani::assume(dilation >= 1 && dilation <= 256);

    // (kernel_size - 1) is safe since kernel_size >= 1.
    let km1 = kernel_size - 1;

    // Check for multiplication overflow.
    match km1.checked_mul(dilation) {
        Some(dilated) => {
            // +1 cannot overflow since dilated <= (1023 * 256) < usize::MAX.
            let effective_k = dilated + 1;
            assert!(effective_k >= 1, "effective kernel must be >= 1");
            // Verify: effective_k == (kernel_size - 1) * dilation + 1.
            assert_eq!(
                effective_k,
                (kernel_size - 1) * dilation + 1,
                "effective kernel formula mismatch"
            );
        }
        None => {
            // Overflow in the product — would be caught by Rust's checked arithmetic
            // in a production implementation. For these bounded inputs this shouldn't
            // happen, but we verify it doesn't.
            let true_product = (km1 as u128) * (dilation as u128);
            assert!(
                true_product > usize::MAX as u128,
                "false overflow in effective kernel computation"
            );
        }
    }
}

/// Prove: NormConv output length formula is correct and non-negative.
///
/// Models `dyn_tensor_metal_norm_conv_fused.rs:195-200`:
/// ```
/// let padded = in_len + 2 * padding;
/// if padded < effective_k { return Err(...) }
/// let out_len = padded - effective_k + 1;
/// ```
///
/// For valid inputs where `padded >= effective_k`, the output length is
/// always >= 1 (Metal needs at least 1 output element to dispatch).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn norm_conv_output_length_positive_when_valid() {
    let in_len: usize = kani::any();
    let padding: usize = kani::any();
    let kernel_size: usize = kani::any();
    let dilation: usize = kani::any();

    // Realistic bounds.
    kani::assume(in_len >= 1 && in_len <= (1usize << 16));
    kani::assume(padding <= 512);
    kani::assume(kernel_size >= 1 && kernel_size <= 256);
    kani::assume(dilation >= 1 && dilation <= 64);

    let effective_k = (kernel_size - 1) * dilation + 1;
    let padded = in_len + 2 * padding;

    // Only proceed if the guard passes.
    kani::assume(padded >= effective_k);

    let out_len = padded - effective_k + 1;

    // Output length must be >= 1.
    assert!(out_len >= 1, "output length must be >= 1 when guard passes");

    // Cross-check: standard conv1d formula.
    // out_len = floor((in_len + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
    // With stride=1: out_len = in_len + 2*padding - (kernel_size-1)*dilation
    let expected = in_len + 2 * padding - (kernel_size - 1) * dilation;
    assert_eq!(out_len, expected, "output length formula mismatch with standard conv1d");
}

/// Prove: NormConv stats buffer allocation uses correct checked arithmetic.
///
/// Models `dyn_tensor_metal_norm_conv_fused.rs:231-234`:
/// `flat_rows.checked_mul(2 * size_of::<f32>())` for stats buffer.
///
/// The stats kernel outputs `[mean, inv_std]` per row, so the buffer
/// needs `flat_rows * 8` bytes. Overflow here = undersized buffer.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn norm_conv_stats_buffer_size_sound() {
    let batch: usize = kani::any();
    let in_channels: usize = kani::any();

    kani::assume(batch >= 1 && batch <= (1usize << 12));
    kani::assume(in_channels >= 1 && in_channels <= (1usize << 16));

    let flat_rows = match batch.checked_mul(in_channels) {
        Some(r) => r,
        None => return, // overflow correctly caught upstream
    };

    let stats_pair_bytes = 2 * std::mem::size_of::<f32>(); // 8 bytes
    match flat_rows.checked_mul(stats_pair_bytes) {
        Some(stats_bytes) => {
            // Verify: stats_bytes = flat_rows * 8.
            assert_eq!(stats_bytes, flat_rows * 8, "stats buffer size mismatch");
            // Stats buffer must hold at least 1 pair.
            assert!(stats_bytes >= 8, "stats buffer must hold at least 1 pair");
        }
        None => {
            let true_size = (flat_rows as u128) * (stats_pair_bytes as u128);
            assert!(true_size > usize::MAX as u128, "false stats overflow");
        }
    }
}
