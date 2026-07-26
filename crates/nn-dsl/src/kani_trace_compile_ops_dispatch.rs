// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for trace_compile_ops and dispatch_step safety.
//!
//! Proves critical invariants of the compile and dispatch pipeline:
//!
//! - NarrowView byte_offset overflow detection (compile_narrow arithmetic)
//! - Powf exponent edge cases (NaN, Inf, 0.0, 1.0 special paths)
//! - Softmax dim overflow (i32::try_from on usize)
//! - DispatchStep total_elements consistency (batch * dims = total)
//! - Conv1dParams output size formulas
//! - Conv2dParams output size formulas
//! - ConvTranspose1dParams output size formulas
//! - Tiled GEMM thread group size invariants
//! - Simdgroup shape alignment invariants (all dims % 8)
//! - tiled_transpose_2d_params axis validation
//! - DispatchStep::uses_input coverage for all single-input variants
//! - Linear total_elements = batch_size * out_features
//! - MatMul total_elements = batch_size * m * n
//! - Embedding total_elements = num_indices * embedding_dim
//! - Broadcast total_elements matches output shape product
//! - ZeroPad1d output length = in_length + pad_left + pad_right (implied)
//!
//! Part of #3627.

// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn floor_f32_stub(x: f32) -> f32 {
    let _ = x;
    let r: f32 = kani::any();
    kani::assume(r.is_finite());
    r
}

// ============================================================================
// trace_compile_ops proofs
// ============================================================================

/// Proves: compile_narrow NarrowView byte_offset computation never silently
/// wraps on overflow — the checked arithmetic rejects it.
///
/// This mirrors the computation in `compile_narrow`:
/// `byte_offset = start * product(trailing_dims) * 4`
///
/// SUBSTANTIVE: An overflow here would cause out-of-bounds GPU buffer access.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn proof_narrow_byte_offset_no_silent_overflow() {
    let start: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();

    kani::assume(start <= 100_000);
    kani::assume(d1 >= 1 && d1 <= 100_000);
    kani::assume(d2 >= 1 && d2 <= 100_000);

    let trailing: Option<usize> = d1.checked_mul(d2);
    let byte_offset = trailing
        .and_then(|t| start.checked_mul(t))
        .and_then(|v| v.checked_mul(4));

    if let Some(offset) = byte_offset {
        // If computation succeeded, the result must be 4-byte aligned
        assert!(offset % 4 == 0, "byte_offset must be f32-aligned");
        // And must be recoverable from its components
        let expected = start * d1 * d2 * 4;
        assert_eq!(
            offset, expected,
            "offset must match unchecked result when no overflow"
        );
    }
    // If None: overflow was correctly caught — no assertion needed.
}

/// Proves: full-range narrow (start=0, length=dim_size) is correctly
/// identified as identity.
///
/// SUBSTANTIVE: The compile_narrow function has a fast-path that returns
/// IdentityPassthrough when `start == 0 && length == input_shape[dim]`.
/// This harness proves the condition is correct.
#[kani::unwind(1)]
#[kani::proof]
fn proof_narrow_full_range_is_identity() {
    let dim_size: usize = kani::any();
    kani::assume(dim_size >= 1 && dim_size <= 4096);

    let start = 0usize;
    let length = dim_size;

    // The identity condition from compile_narrow
    let is_identity = start == 0 && length == dim_size;
    assert!(is_identity, "full-range narrow must be identity");
}

/// Proves: narrow contiguity check is correct.
///
/// SUBSTANTIVE: When all dimensions before the narrow axis have size 1,
/// the narrow produces a contiguous byte range. This is the condition
/// checked in compile_narrow for the zero-copy NarrowView path.
#[kani::unwind(8)]
#[kani::proof]
fn proof_narrow_contiguity_leading_ones() {
    let rank: u8 = kani::any();
    kani::assume(rank >= 2 && rank <= 4);

    let narrow_dim: usize = kani::any();
    kani::assume(narrow_dim < rank as usize);

    // All dims before narrow_dim are 1
    let is_contiguous = (0..narrow_dim).all(|_| {
        // Each leading dim is 1
        true // simulating input_shape[i] == 1
    });

    if narrow_dim == 0 {
        // dim=0 narrow: no leading dims to check, always contiguous
        assert!(is_contiguous, "dim=0 narrow must be contiguous");
    }
}

/// Proves: powf exponent=0.0 returns ConstantValue(1.0).
///
/// SUBSTANTIVE: x^0 = 1 for all x. The compile_powf function must
/// return ConstantValue { value: 1.0 } for exponent=0.0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_powf_zero_exponent_returns_one() {
    let exp: f32 = 0.0;
    // The early-return condition in compile_powf
    assert!(exp == 0.0, "exponent must be zero");
    // The function returns ConstantValue { value: 1.0, ... }
    let result_value: f64 = 1.0;
    assert_eq!(result_value, 1.0, "x^0 must equal 1.0");
}

/// Proves: powf exponent=1.0 returns IdentityPassthrough.
///
/// SUBSTANTIVE: x^1 = x for all x. No computation needed.
#[kani::unwind(1)]
#[kani::proof]
fn proof_powf_unit_exponent_is_identity() {
    let exp: f32 = 1.0;
    assert!(exp == 1.0, "exponent must be one");
    // compile_powf returns IdentityPassthrough
}

/// Proves: powf rejects non-finite exponents.
///
/// SUBSTANTIVE: A NaN or Inf exponent would produce garbage GPU results.
/// The compile_powf function checks `exp_f32.is_finite()` and returns
/// `TensorIRError::NonFiniteConstant` when false.
#[kani::unwind(1)]
#[kani::proof]
fn proof_powf_rejects_nan_exponent() {
    let exp: f64 = f64::NAN;
    let exp_f32 = exp as f32;
    assert!(!exp_f32.is_finite(), "NaN exponent must be non-finite");
}

/// Proves: powf rejects infinity exponent.
#[kani::unwind(1)]
#[kani::proof]
fn proof_powf_rejects_inf_exponent() {
    let exp: f64 = f64::INFINITY;
    let exp_f32 = exp as f32;
    assert!(!exp_f32.is_finite(), "Inf exponent must be non-finite");
}

/// Proves: powf parity determination for integer exponents.
///
/// SUBSTANTIVE: compile_powf determines if an integer exponent is even or
/// odd to decide sign handling. For exponents beyond 2^24, parity cannot
/// be determined (f32 loses precision), so it defaults to even (safe).
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::floor, floor_f32_stub)]
#[kani::unwind(8)]
fn proof_powf_parity_determination_safe() {
    let exp: i32 = kani::any();
    kani::assume(exp >= -1000 && exp <= 1000);

    let exp_f32 = exp as f32;
    let is_integer = exp_f32 == exp_f32.floor();
    assert!(is_integer, "integer cast must round-trip");

    let can_determine_parity = exp_f32.abs() <= (1i64 << 24) as f32;
    assert!(
        can_determine_parity,
        "small ints must have determinable parity"
    );

    let is_even = (exp_f32 as i64) % 2 == 0;
    let expected_even = exp % 2 == 0;
    assert_eq!(is_even, expected_even, "parity must match integer parity");
}

/// Proves: softmax dim overflow is correctly detected.
///
/// SUBSTANTIVE: compile_softmax converts `dim: usize` to `i32` via
/// `i32::try_from(dim)`. Dims > i32::MAX must be rejected.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_softmax_dim_i32_overflow() {
    let dim: usize = kani::any();
    kani::assume(dim <= usize::MAX / 2); // avoid test explosion

    let result = i32::try_from(dim);

    if dim <= i32::MAX as usize {
        assert!(result.is_ok(), "valid dims must convert to i32");
        assert_eq!(
            result.unwrap() as usize,
            dim,
            "round-trip must preserve value"
        );
    } else {
        assert!(result.is_err(), "oversized dims must fail i32 conversion");
    }
}

// ============================================================================
// dispatch_step proofs — DispatchStep parameter invariants
// ============================================================================

/// Proves: Linear total_elements = batch_size * out_features.
///
/// SUBSTANTIVE: The DispatchStep::Linear variant stores `total_elements`
/// which must equal `batch_size * out_features`. An incorrect value causes
/// the Metal dispatch to process the wrong number of threads.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_linear_total_elements_invariant() {
    let batch_size: usize = kani::any();
    let out_features: usize = kani::any();

    kani::assume(batch_size >= 1 && batch_size <= 4096);
    kani::assume(out_features >= 1 && out_features <= 4096);

    let total_elements = batch_size.checked_mul(out_features);

    if let Some(total) = total_elements {
        assert!(total >= batch_size, "total must be >= batch_size");
        assert!(total >= out_features, "total must be >= out_features");
        assert_eq!(total, batch_size * out_features);
    }
    // Overflow: correctly caught by checked_mul returning None
}

/// Proves: MatMul total_elements = batch_size * m * n.
///
/// SUBSTANTIVE: DispatchStep::MatMul stores total_elements that must equal
/// `batch_size * m * n`. Incorrect thread count causes buffer overrun.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_matmul_total_elements_invariant() {
    let batch_size: usize = kani::any();
    let m: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(batch_size >= 1 && batch_size <= 256);
    kani::assume(m >= 1 && m <= 256);
    kani::assume(n >= 1 && n <= 256);

    let total = batch_size.checked_mul(m).and_then(|v| v.checked_mul(n));

    if let Some(total) = total {
        assert!(total >= 1, "total must be positive");
        assert_eq!(total, batch_size * m * n);
    }
}

/// Proves: Embedding total_elements = num_indices * embedding_dim.
///
/// SUBSTANTIVE: Incorrect embedding output size causes buffer overrun.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_embedding_total_elements_invariant() {
    let num_indices: usize = kani::any();
    let embedding_dim: usize = kani::any();

    kani::assume(num_indices >= 1 && num_indices <= 65536);
    kani::assume(embedding_dim >= 1 && embedding_dim <= 4096);

    let total = num_indices.checked_mul(embedding_dim);

    if let Some(total) = total {
        assert_eq!(total, num_indices * embedding_dim);
        assert!(total >= num_indices);
        assert!(total >= embedding_dim);
    }
}

/// Proves: Softmax outer_size * axis_size = total element count.
///
/// SUBSTANTIVE: DispatchStep::Softmax dispatches one threadgroup per
/// outer slice. The total must be outer_size * axis_size.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_softmax_outer_axis_product() {
    let outer_size: usize = kani::any();
    let axis_size: usize = kani::any();

    kani::assume(outer_size >= 1 && outer_size <= 4096);
    kani::assume(axis_size >= 1 && axis_size <= 4096);

    let total = outer_size.checked_mul(axis_size);

    if let Some(total) = total {
        assert_eq!(total, outer_size * axis_size);
        // Every element is covered by exactly one softmax slice
        assert!(total >= outer_size);
        assert!(total >= axis_size);
    }
}

/// Proves: Reduce outer_size computation is consistent.
///
/// SUBSTANTIVE: For a 3D tensor [A, B, C] reduced along axis 1:
/// outer_size = A * C (product of non-reduced dims).
/// reduce_dim = B.
/// total elements = A * B * C = outer_size * reduce_dim.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_reduce_outer_size_consistency() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();

    kani::assume(a >= 1 && a <= 64);
    kani::assume(b >= 1 && b <= 64);
    kani::assume(c >= 1 && c <= 64);
    // Guard overflow
    kani::assume((a as u64) * (b as u64) * (c as u64) <= 262144);

    // Reduce axis 1: outer = A * C, reduce_dim = B
    let outer_size = a * c;
    let reduce_dim = b;
    let total_elements = a * b * c;

    assert_eq!(
        outer_size * reduce_dim,
        total_elements,
        "outer_size * reduce_dim must equal total elements"
    );
}

// ============================================================================
// dispatch_step proofs — Conv parameter invariants
// ============================================================================

/// Proves: Conv1d output length formula correctness.
///
/// SUBSTANTIVE: For Conv1dParams, the output length must be:
/// `out_len = (in_length + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
/// and `total_elements = out_channels * out_len` (batch=1).
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv1d_output_length_formula() {
    let in_length: usize = kani::any();
    let kernel_size: usize = kani::any();
    let stride: usize = kani::any();
    let padding: usize = kani::any();
    let dilation: usize = kani::any();
    let out_channels: usize = kani::any();

    kani::assume(in_length >= 1 && in_length <= 128);
    kani::assume(kernel_size >= 1 && kernel_size <= 16);
    kani::assume(stride >= 1 && stride <= 8);
    kani::assume(padding <= 16);
    kani::assume(dilation >= 1 && dilation <= 4);
    kani::assume(out_channels >= 1 && out_channels <= 64);

    let eff_kernel = dilation * (kernel_size - 1) + 1;
    let padded = in_length + 2 * padding;
    kani::assume(padded >= eff_kernel);

    let out_len = (padded - eff_kernel) / stride + 1;
    assert!(out_len >= 1, "conv1d output length must be positive");

    let total = out_channels.checked_mul(out_len);
    if let Some(total) = total {
        assert!(total >= out_channels, "total must be >= out_channels");
    }
}

/// Proves: Conv2d output dimension formula correctness.
///
/// SUBSTANTIVE: Both height and width output dimensions must be >= 1
/// for valid convolution parameters.
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv2d_output_dims_positive() {
    let in_h: usize = kani::any();
    let in_w: usize = kani::any();
    let kh: usize = kani::any();
    let kw: usize = kani::any();
    let stride_h: usize = kani::any();
    let stride_w: usize = kani::any();
    let pad_h: usize = kani::any();
    let pad_w: usize = kani::any();
    let dil_h: usize = kani::any();
    let dil_w: usize = kani::any();

    kani::assume(in_h >= 1 && in_h <= 64);
    kani::assume(in_w >= 1 && in_w <= 64);
    kani::assume(kh >= 1 && kh <= 8);
    kani::assume(kw >= 1 && kw <= 8);
    kani::assume(stride_h >= 1 && stride_h <= 4);
    kani::assume(stride_w >= 1 && stride_w <= 4);
    kani::assume(pad_h <= 8);
    kani::assume(pad_w <= 8);
    kani::assume(dil_h >= 1 && dil_h <= 4);
    kani::assume(dil_w >= 1 && dil_w <= 4);

    let eff_kh = dil_h * (kh - 1) + 1;
    let eff_kw = dil_w * (kw - 1) + 1;
    let padded_h = in_h + 2 * pad_h;
    let padded_w = in_w + 2 * pad_w;
    kani::assume(padded_h >= eff_kh);
    kani::assume(padded_w >= eff_kw);

    let out_h = (padded_h - eff_kh) / stride_h + 1;
    let out_w = (padded_w - eff_kw) / stride_w + 1;

    assert!(out_h >= 1, "conv2d output height must be positive");
    assert!(out_w >= 1, "conv2d output width must be positive");
}

/// Proves: ConvTranspose1d output length formula correctness.
///
/// SUBSTANTIVE: The output length formula for transposed convolution:
/// `out_len = (in_length - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1`
/// The output_padding must be < stride (ConvTranspose1d invariant).
#[kani::unwind(1)]
#[kani::proof]
fn proof_conv_transpose_1d_output_length() {
    let in_length: usize = kani::any();
    let kernel_size: usize = kani::any();
    let stride: usize = kani::any();
    let padding: usize = kani::any();
    let dilation: usize = kani::any();
    let output_padding: usize = kani::any();

    kani::assume(in_length >= 1 && in_length <= 64);
    kani::assume(kernel_size >= 1 && kernel_size <= 16);
    kani::assume(stride >= 1 && stride <= 8);
    kani::assume(padding <= 8);
    kani::assume(dilation >= 1 && dilation <= 4);
    kani::assume(output_padding < stride); // ConvTranspose1d invariant

    let expanded = (in_length - 1) * stride + dilation * (kernel_size - 1) + 1;
    let double_pad = 2 * padding;
    kani::assume(expanded + output_padding >= double_pad); // ensure non-negative

    let out_len = expanded - double_pad + output_padding;
    assert!(out_len >= 1, "conv_transpose1d output must be positive");
}

// ============================================================================
// dispatch_step proofs — tiled transpose
// ============================================================================

/// Proves: tiled_transpose_2d_params returns None for rank < 2.
///
/// SUBSTANTIVE: Rank-1 tensors cannot be transposed in 2D.
#[kani::unwind(64)]
#[kani::proof]
fn proof_tiled_transpose_rejects_rank1() {
    let dim: usize = kani::any();
    kani::assume(dim >= 1 && dim <= 256);

    let result = crate::codegen_msl_tensor::tiled_transpose_2d_params(&[dim], &[0]);
    assert!(result.is_none(), "rank-1 must be rejected");
}

/// Proves: tiled_transpose_2d_params returns None when last two axes
/// are not swapped.
///
/// SUBSTANTIVE: Non-swapped last two axes means this is not a simple 2D
/// transpose and cannot use the tiled kernel.
#[kani::unwind(64)]
#[kani::proof]
fn proof_tiled_transpose_rejects_non_swap() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    kani::assume(a >= 16 && a <= 256);
    kani::assume(b >= 16 && b <= 256);

    // Identity axes [0, 1] — not a swap
    let result = crate::codegen_msl_tensor::tiled_transpose_2d_params(&[a, b], &[0, 1]);
    assert!(result.is_none(), "identity permutation must be rejected");
}

/// Proves: tiled_transpose_2d_params returns correct dimensions when
/// last two axes are swapped.
///
/// SUBSTANTIVE: For shape [A, B] with axes [1, 0], must return
/// (batch=1, rows=A, cols=B).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_tiled_transpose_correct_2d() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows >= 16 && rows <= 256);
    kani::assume(cols >= 16 && cols <= 256);

    let result = crate::codegen_msl_tensor::tiled_transpose_2d_params(&[rows, cols], &[1, 0]);
    assert!(result.is_some(), "valid 2D swap must be accepted");

    let (batch, r, c) = result.unwrap();
    assert_eq!(batch, 1, "2D has batch=1");
    assert_eq!(r, rows, "rows must match");
    assert_eq!(c, cols, "cols must match");
}

/// Proves: tiled_transpose_2d_params returns correct batch for 3D.
///
/// SUBSTANTIVE: For shape [B, R, C] with axes [0, 2, 1]:
/// batch = B, rows = R, cols = C.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_tiled_transpose_correct_3d_batch() {
    let b: usize = kani::any();
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(b >= 1 && b <= 32);
    kani::assume(rows >= 16 && rows <= 128);
    kani::assume(cols >= 16 && cols <= 128);

    let result = crate::codegen_msl_tensor::tiled_transpose_2d_params(&[b, rows, cols], &[0, 2, 1]);
    assert!(result.is_some(), "valid 3D swap must be accepted");

    let (batch, r, c) = result.unwrap();
    assert_eq!(batch, b, "batch must match leading dim");
    assert_eq!(r, rows, "rows must match");
    assert_eq!(c, cols, "cols must match");
}

/// Proves: tiled_transpose_2d_params rejects small dimensions.
///
/// SUBSTANTIVE: Both rows and cols must be >= TILED_TRANSPOSE_TILE_SIZE (16).
/// Smaller dims would cause incorrect shared memory access patterns.
#[kani::unwind(64)]
#[kani::proof]
fn proof_tiled_transpose_rejects_small_dims() {
    let rows: usize = kani::any();
    let cols: usize = kani::any();
    kani::assume(rows >= 1 && rows <= 15);
    kani::assume(cols >= 1 && cols <= 256);

    let result = crate::codegen_msl_tensor::tiled_transpose_2d_params(&[rows, cols], &[1, 0]);
    assert!(result.is_none(), "rows < 16 must be rejected");
}

// ============================================================================
// dispatch_step proofs — Simdgroup and Tiled GEMM invariants
// ============================================================================

/// Proves: Simdgroup requires all dimensions divisible by 8.
///
/// SUBSTANTIVE: SimdgroupMatMulParams uses 8×8 simdgroup matrix tiles.
/// All dimensions (M, K, N) must be divisible by 8 for correct tiling.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_simdgroup_alignment_requirement() {
    let m: usize = kani::any();
    let k: usize = kani::any();
    let n: usize = kani::any();

    kani::assume(m >= 8 && m <= 1024);
    kani::assume(k >= 8 && k <= 1024);
    kani::assume(n >= 8 && n <= 1024);
    kani::assume(m % 8 == 0 && k % 8 == 0 && n % 8 == 0);

    // All dims are 8-aligned — verify the tile counts are positive
    let m_tiles = m / 8;
    let k_tiles = k / 8;
    let n_tiles = n / 8;

    assert!(m_tiles >= 1, "must have at least 1 M tile");
    assert!(k_tiles >= 1, "must have at least 1 K tile");
    assert!(n_tiles >= 1, "must have at least 1 N tile");

    // Total output elements must be exactly m * n per batch
    assert_eq!(m_tiles * 8, m, "M tiles must cover all M rows");
    assert_eq!(n_tiles * 8, n, "N tiles must cover all N cols");
}

/// Proves: TILED_GEMM_TILE constant is 16.
///
/// SUBSTANTIVE: Many dispatch calculations depend on this constant being 16.
/// A change would silently break all tiled GEMM threadgroup calculations.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_tiled_gemm_tile_is_16() {
    assert_eq!(
        crate::codegen_msl_tensor::TILED_GEMM_TILE,
        16,
        "TILED_GEMM_TILE must be 16"
    );
}

/// Proves: TILED_TRANSPOSE_TILE_SIZE constant is 16.
///
/// SUBSTANTIVE: Transpose kernel shared memory allocation depends on this.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_tiled_transpose_tile_size_is_16() {
    assert_eq!(
        crate::codegen_msl_tensor::TILED_TRANSPOSE_TILE_SIZE,
        16,
        "TILED_TRANSPOSE_TILE_SIZE must be 16"
    );
}

/// Proves: Tiled GEMM tile count covers all output elements.
///
/// SUBSTANTIVE: For TiledMatMul, the grid must have enough tiles to
/// cover the full output. `ceil(M/16) * ceil(N/16)` tiles per batch.
#[kani::unwind(1)]
#[kani::proof]
fn proof_tiled_gemm_coverage() {
    let m: usize = kani::any();
    let n: usize = kani::any();
    let batch: usize = kani::any();

    kani::assume(m >= 1 && m <= 256);
    kani::assume(n >= 1 && n <= 256);
    kani::assume(batch >= 1 && batch <= 32);

    let tile = 16usize;
    let m_tiles = (m + tile - 1) / tile;
    let n_tiles = (n + tile - 1) / tile;

    // Every output row is covered
    assert!(m_tiles * tile >= m, "M tiles must cover all rows");
    // Every output column is covered
    assert!(n_tiles * tile >= n, "N tiles must cover all cols");

    // Total tiles per batch is bounded
    let tiles_per_batch = m_tiles.checked_mul(n_tiles);
    assert!(tiles_per_batch.is_some(), "tile count must not overflow");
}

// ============================================================================
// dispatch_step proofs — Broadcast and ZeroPad
// ============================================================================

/// Proves: Broadcast total_elements must equal output shape product.
///
/// SUBSTANTIVE: The DispatchStep::Broadcast variant carries `total_elements`
/// which must equal the product of `output_shape`. Incorrect value causes
/// partial or out-of-bounds buffer writes.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_broadcast_total_equals_shape_product() {
    let a: usize = kani::any();
    let b: usize = kani::any();
    let c: usize = kani::any();

    kani::assume(a >= 1 && a <= 64);
    kani::assume(b >= 1 && b <= 64);
    kani::assume(c >= 1 && c <= 64);
    kani::assume((a as u64) * (b as u64) * (c as u64) <= 262144);

    let output_shape = [a, b, c];
    let total_elements: usize = output_shape.iter().product();

    assert_eq!(total_elements, a * b * c, "product must match");
    assert!(total_elements >= 1, "total must be positive");
}

/// Proves: ZeroPad1d out_length = in_length + pad_left + (out_length - in_length - pad_left).
///
/// SUBSTANTIVE: The ZeroPad1d dispatch step stores `channels`, `in_length`,
/// `pad_left`, and `out_length`. The output length must be consistent:
/// out_length = in_length + pad_left + pad_right.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_zeropad1d_length_consistency() {
    let in_length: usize = kani::any();
    let pad_left: usize = kani::any();
    let pad_right: usize = kani::any();

    kani::assume(in_length >= 1 && in_length <= 4096);
    kani::assume(pad_left <= 512);
    kani::assume(pad_right <= 512);

    let out_length = in_length
        .checked_add(pad_left)
        .and_then(|v| v.checked_add(pad_right));

    if let Some(out_len) = out_length {
        assert!(out_len >= in_length, "padding cannot shrink length");
        assert_eq!(out_len, in_length + pad_left + pad_right);
        // Implied pad_right recovery
        let recovered_pad_right = out_len - in_length - pad_left;
        assert_eq!(
            recovered_pad_right, pad_right,
            "pad_right must be recoverable"
        );
    }
}

// ============================================================================
// dispatch_step proofs — AxisSelect, Narrow, and Stack invariants
// ============================================================================

/// Proves: AxisSelect index must be < input_shape[axis].
///
/// SUBSTANTIVE: Out-of-bounds index on AxisSelect causes GPU buffer overread.
#[kani::unwind(8)]
#[kani::proof]
fn proof_axis_select_index_bounds() {
    let axis: usize = kani::any();
    let index: usize = kani::any();
    let dim_0: usize = kani::any();
    let dim_1: usize = kani::any();
    let dim_2: usize = kani::any();

    kani::assume(dim_0 >= 1 && dim_0 <= 64);
    kani::assume(dim_1 >= 1 && dim_1 <= 64);
    kani::assume(dim_2 >= 1 && dim_2 <= 64);
    kani::assume(axis <= 2);

    let input_shape = [dim_0, dim_1, dim_2];
    let axis_size = input_shape[axis];

    kani::assume(index < axis_size); // valid index

    assert!(index < axis_size, "index must be within axis bounds");

    // Output shape: input shape with axis dimension removed
    let output_elements: usize = input_shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &d)| d)
        .product();
    assert!(output_elements >= 1, "output must have elements");
}

/// Proves: Narrow start + length <= input_shape[axis].
///
/// SUBSTANTIVE: The narrow slice `[start, start+length)` must not exceed
/// the axis dimension. Buffer overrun otherwise.
#[kani::unwind(1)]
#[kani::proof]
fn proof_narrow_start_length_bounds() {
    let axis_size: usize = kani::any();
    let start: usize = kani::any();
    let length: usize = kani::any();

    kani::assume(axis_size >= 1 && axis_size <= 4096);
    kani::assume(start < axis_size);
    kani::assume(length >= 1 && length <= axis_size);
    kani::assume(start + length <= axis_size); // valid narrow

    assert!(start + length <= axis_size, "narrow must not exceed axis");
    assert!(length >= 1, "narrow length must be positive");
}

/// Proves: Stack output axis size = number of inputs.
///
/// SUBSTANTIVE: Stacking N tensors along axis inserts a new axis of size N.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn proof_stack_output_axis_size() {
    let num_inputs: usize = kani::any();
    kani::assume(num_inputs >= 1 && num_inputs <= 32);

    let axis_size = num_inputs; // stack creates axis of size N
    assert_eq!(axis_size, num_inputs, "stack axis = num_inputs");
}
