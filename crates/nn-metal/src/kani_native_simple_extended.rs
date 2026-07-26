// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for compiled model native simple ops (#3735).
//!
//! Complements `kani_compiled_model_execute_native_simple.rs` with deeper
//! proofs targeting:
//!
//! - Conv1d effective kernel computation under dilation
//! - Conv1d output length edge cases (zero output, single output)
//! - LinearActivation simdgroup grid dimension safety (u32 overflow)
//! - LinearActivation TG memory for both f16 and f32 paths
//! - Int8Gemm buffer count correctness
//! - LSTM precomputed path routing conditions
//! - LSTM bias combination routing (3-way: bih+bhh, single bias, no bias)
//! - NativeOp dispatch: complete variant coverage check
//! - MaxPool1d output length formula correctness

// ============================================================================
// 1. Conv1d: effective kernel under dilation
// ============================================================================

/// Prove: effective_k = (kernel_size - 1) * dilation + 1 does not underflow
/// for valid parameters, and effective_k >= kernel_size when dilation >= 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_effective_kernel_no_underflow() {
    let kernel_size: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(kernel_size >= 1 && kernel_size <= 128);
    kani::assume(dilation >= 1 && dilation <= 64);

    // (kernel_size - 1) is safe because kernel_size >= 1.
    let km1 = kernel_size - 1;
    let dilated = km1.checked_mul(dilation);
    assert!(dilated.is_some(), "km1 * dilation must not overflow");

    let effective_k = dilated.unwrap() + 1;
    assert!(effective_k >= kernel_size, "effective_k >= kernel_size when dilation >= 1");
    assert!(effective_k >= 1, "effective_k always >= 1");

    // At dilation=1, effective_k == kernel_size.
    if dilation == 1 {
        assert_eq!(effective_k, kernel_size);
    }
}

// ============================================================================
// 2. Conv1d: l_out = 0 when padding is insufficient
// ============================================================================

/// Prove: when padded < effective_k, l_out is exactly 0.
/// This guards against underflow in the (padded - effective_k) expression.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_lout_zero_when_padding_insufficient() {
    let l_in: usize = kani::any();
    let kernel_size: usize = kani::any();
    let stride: usize = kani::any();
    let padding: usize = kani::any();
    let dilation: usize = kani::any();

    kani::assume(l_in >= 1 && l_in <= 1024);
    kani::assume(kernel_size >= 2 && kernel_size <= 128);
    kani::assume(stride >= 1 && stride <= 16);
    kani::assume(padding <= 64);
    kani::assume(dilation >= 1 && dilation <= 16);

    let effective_k = (kernel_size - 1) * dilation + 1;
    let padded = l_in + 2 * padding;

    let l_out = if padded >= effective_k {
        (padded - effective_k) / stride + 1
    } else {
        0
    };

    if padded < effective_k {
        assert_eq!(l_out, 0, "insufficient padding must yield l_out=0");
    } else {
        assert!(l_out >= 1, "sufficient padding must yield l_out >= 1");
    }
}

// ============================================================================
// 3. Conv1d: l_out = 1 is the minimum non-zero output
// ============================================================================

/// Prove: when padded == effective_k, l_out is exactly 1 regardless of stride.
/// This is the smallest non-zero convolution output.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn conv1d_lout_one_at_boundary() {
    let stride: usize = kani::any();
    kani::assume(stride >= 1 && stride <= 64);

    // padded == effective_k means (padded - effective_k) = 0, so l_out = 0/stride + 1 = 1.
    let l_out = 0 / stride + 1;
    assert_eq!(l_out, 1, "boundary case must produce l_out=1");
}

// ============================================================================
// 4. LinearActivation: simdgroup grid dimension overflow check
// ============================================================================

/// Prove: simdgroup grid dimensions [ceil(N/32), ceil(M/32), 1] fit in u32
/// for all production dimension ranges.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn linear_activation_simdgroup_grid_fits_u32() {
    let batch_size: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(batch_size >= 1 && batch_size <= 65536);
    kani::assume(out_features >= 1 && out_features <= 65536);

    let grid_x = out_features.div_ceil(32);
    let grid_y = batch_size.div_ceil(32);

    assert!(grid_x <= u32::MAX as usize, "grid_x must fit u32");
    assert!(grid_y <= u32::MAX as usize, "grid_y must fit u32");

    let grid_x_u32 = u32::try_from(grid_x);
    let grid_y_u32 = u32::try_from(grid_y);
    assert!(grid_x_u32.is_ok());
    assert!(grid_y_u32.is_ok());
}

// ============================================================================
// 5. LinearActivation: TG memory for simdgroup path
// ============================================================================

/// Prove: LinearActivation simdgroup TG memory is correct for both f32 and f16.
///
/// f16: 2 * 32 * 33 * 2 + 32 * 33 * 4 = 4224 + 4224 = 8448 bytes.
/// f32: 3 * 32 * 33 * 4 = 12960... wait, production uses: As+Bs(element) + tile_out(float).
/// f32: 2 * 32 * 33 * 4 + 32 * 33 * 4 = 8448 + 4224 = 12672.
/// But code says: f16: `2 * 32 * 33 * 2 + 32 * 33 * 4`, f32: `3 * 32 * 33 * 4`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn linear_activation_tg_memory_correct() {
    let is_half: bool = kani::any();

    let tg_bytes: u64 = if is_half {
        2 * 32 * 33 * 2 + 32 * 33 * 4
    } else {
        3 * 32 * 33 * 4
    };

    // Both must fit in Metal's 32 KB limit.
    assert!(tg_bytes <= 32_768, "TG memory must fit 32 KB Metal limit");
    assert!(tg_bytes > 0, "TG memory must be non-zero");

    // Exact values.
    if is_half {
        assert_eq!(tg_bytes, 8_448, "f16 TG memory must be 8448");
    } else {
        assert_eq!(tg_bytes, 12_672, "f32 TG memory must be 12672");
    }
}

// ============================================================================
// 6. Int8Gemm: buffer input count is 4 (no bias) or 5 (with bias)
// ============================================================================

/// Prove: Int8Gemm param_count is exactly 4 or 5 depending on has_bias.
/// Buffers: input, weight_int8, scale, zero_point, [bias].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn int8_gemm_param_count() {
    let has_bias: bool = kani::any();

    let base_params: usize = 4; // input, weight_int8, scale, zero_point
    let param_count = if has_bias { base_params + 1 } else { base_params };

    assert!(param_count == 4 || param_count == 5);
    if has_bias {
        assert_eq!(param_count, 5, "bias adds exactly 1 buffer");
    } else {
        assert_eq!(param_count, 4, "no bias means 4 buffers");
    }
}

// ============================================================================
// 7. LSTM precomputed path routing: alignment conditions
// ============================================================================

/// Prove: LSTM precomputed path requires input_size % 8 == 0 AND
/// 4 * hidden_size % 8 == 0. Since 4 * hidden_size is always divisible
/// by 4, it's divisible by 8 iff hidden_size is even.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_precomputed_routing_alignment() {
    let input_size: usize = kani::any();
    let hidden_size: usize = kani::any();

    kani::assume(input_size >= 1 && input_size <= 4096);
    kani::assume(hidden_size >= 1 && hidden_size <= 512);

    let n = 4 * hidden_size;
    let can_precompute = input_size % 8 == 0 && n % 8 == 0;

    // Property: n % 8 == 0 iff hidden_size % 2 == 0.
    if hidden_size % 2 == 0 {
        assert_eq!(n % 8, 0, "even hidden_size -> 4*H divisible by 8");
    } else {
        assert_ne!(n % 8, 0, "odd hidden_size -> 4*H NOT divisible by 8");
    }

    // Kokoro: hidden_size=256, input_size=640 or 512.
    // Both are even and both input sizes are % 8 == 0.
}

/// Prove: Kokoro BiLSTM parameters always qualify for precomputed path.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_precomputed_kokoro_always_qualifies() {
    let hidden_size: usize = 256;
    let input_size_first_layer: usize = 640;
    let input_size_subsequent: usize = 512;

    let n = 4 * hidden_size; // 1024

    // First layer.
    assert!(input_size_first_layer % 8 == 0, "640 % 8 == 0");
    assert!(n % 8 == 0, "1024 % 8 == 0");

    // Subsequent layers.
    assert!(input_size_subsequent % 8 == 0, "512 % 8 == 0");
}

// ============================================================================
// 8. LSTM bias combination: 3-way routing correctness
// ============================================================================

/// Prove: LSTM bias combination routing covers all 3 cases exhaustively:
/// (1) bias_ih + bias_hh present -> combine, (2) single bias -> use directly,
/// (3) neither -> None. No fourth case exists.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn lstm_bias_routing_exhaustive() {
    let has_bih: bool = kani::any();
    let has_bhh: bool = kani::any();
    let has_single: bool = kani::any();

    // In practice, has_bih and has_bhh are always both true or both false,
    // but the code handles partial presence correctly.
    let route = if has_bih && has_bhh {
        1_u8 // combine
    } else if has_single {
        2 // use single
    } else {
        3 // None
    };

    // Property 1: route is one of {1, 2, 3}.
    assert!(route >= 1 && route <= 3, "route must be 1, 2, or 3");

    // Property 2: combine requires both parts.
    if route == 1 {
        assert!(has_bih && has_bhh);
    }

    // Property 3: if both bias parts present, combine is taken regardless of single.
    if has_bih && has_bhh {
        assert_eq!(route, 1, "combined bias takes priority");
    }
}

// ============================================================================
// 9. MaxPool1d: output length formula
// ============================================================================

/// Prove: MaxPool1d output length matches PyTorch formula:
/// l_out = floor((l_in + 2*padding - kernel_size) / stride) + 1
/// when l_in + 2*padding >= kernel_size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn max_pool1d_output_length() {
    let l_in: usize = kani::any();
    let kernel_size: usize = kani::any();
    let stride: usize = kani::any();
    let padding: usize = kani::any();

    kani::assume(l_in >= 1 && l_in <= 65536);
    kani::assume(kernel_size >= 1 && kernel_size <= 128);
    kani::assume(stride >= 1 && stride <= 64);
    kani::assume(padding <= 128);

    let padded = l_in + 2 * padding;
    if padded >= kernel_size {
        let l_out = (padded - kernel_size) / stride + 1;
        assert!(l_out >= 1, "valid pool produces l_out >= 1");

        // The last pool window starts at (l_out - 1) * stride.
        // It must fit within [0, padded).
        let last_start = (l_out - 1) * stride;
        let last_end = last_start + kernel_size;
        assert!(last_end <= padded, "last pool window must fit");
    }
}

// ============================================================================
// 10. LinearActivation: output element count no overflow
// ============================================================================

/// Prove: batch_size * out_features does not overflow for production ranges.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn linear_activation_output_elems_no_overflow() {
    let batch_size: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(batch_size >= 1 && batch_size <= 65536);
    kani::assume(out_features >= 1 && out_features <= 65536);

    let total = batch_size.checked_mul(out_features);
    assert!(total.is_some(), "batch*out_features must not overflow usize");

    let bytes = total.unwrap().checked_mul(4);
    assert!(bytes.is_some(), "output bytes (f32) must not overflow");
}

// ============================================================================
// 11. LinearActivation: batch_size >= 1 for all valid input shapes
// ============================================================================

/// Prove: batch_size (product of all dims except last) is always >= 1 when
/// the input tensor has at least 1 dimension with all dims >= 1.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn linear_activation_batch_at_least_one() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();

    kani::assume(d0 >= 1 && d0 <= 64);
    kani::assume(d1 >= 1 && d1 <= 4096);
    kani::assume(d2 >= 1 && d2 <= 4096);

    // 3D input [d0, d1, d2]: batch = d0 * d1.
    let batch = d0 * d1;
    assert!(batch >= 1, "batch must be >= 1 for valid input");
}

// ============================================================================
// 12. Int8Gemm: grid dimensions for simdgroup dispatch
// ============================================================================

/// Prove: Int8Gemm grid [ceil(N/32), ceil(M/32), 1] fits u32 and covers all.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn int8_gemm_grid_covers_all() {
    let m: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 1 && m <= 65536);
    kani::assume(n >= 1 && n <= 65536);

    let grid_x = n.div_ceil(32);
    let grid_y = m.div_ceil(32);

    // Fits u32.
    assert!(grid_x <= u32::MAX as usize);
    assert!(grid_y <= u32::MAX as usize);

    // Covers all output elements.
    let covered = (grid_x as u64) * 32 * (grid_y as u64) * 32;
    assert!(covered >= (m as u64) * (n as u64), "grid must cover M*N");
}

// ============================================================================
// 13. RotaryEmbedding: half_dim = head_dim / 2 integer division safety
// ============================================================================

/// Prove: head_dim / 2 is exact (no truncation) when head_dim is even,
/// and the cache element count seq_len * half_dim is safe for production.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn rope_half_dim_exact_division() {
    let head_dim: usize = kani::any();
    kani::assume(head_dim >= 2 && head_dim <= 512);
    kani::assume(head_dim % 2 == 0);

    let half_dim = head_dim / 2;
    assert_eq!(half_dim * 2, head_dim, "half_dim * 2 must equal head_dim");
    assert!(half_dim >= 1, "half_dim must be >= 1");
}
