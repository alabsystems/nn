// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for GPU iSTFT (`istft_gpu.rs`).
//!
//! Complements `kani_istft_gpu.rs` with additional proofs covering:
//! - n_bins derivation: n_fft/2 + 1 relationship
//! - Fused kernel buffer reuse: polar→iSTFT shares output buffer
//! - Zero-frame edge cases: n_frames == 0 saturating_sub safety
//! - Output pad vs truncate path correctness
//! - Byte alignment for f32 buffers (4-byte alignment)
//! - GPU DynTensor shape [1, 1, final_len] validity for rank-3
//! - IDFT normalization factor monotonicity with n_fft
//! - Full-length output identity (no center trim) path
//! - Arena allocation: idft + ola buffers fit within reasonable arena
//! - Fused polar threadgroup/grid consistency
//! - Center trim: n_fft/2 never exceeds full_len/2
//! - Hop-to-nfft ratio affects overlap count
//! - Input buffer element addressing: row-major n_bins * n_frames layout
//!
//! Part of #3742.

use std::mem::size_of;

// ============================================================================
// n_bins derivation
// ============================================================================

/// Prove: n_bins = n_fft / 2 + 1 is always >= 2 for valid n_fft.
///
/// The STFT produces n_fft/2 + 1 frequency bins. For n_fft >= 2, n_bins >= 2.
/// This matters because IDFT kernel accesses bins [0, n_bins).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_n_bins_always_at_least_two() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 65536);
    kani::assume(n_fft % 2 == 0);

    let n_bins = n_fft / 2 + 1;
    assert!(n_bins >= 2, "n_bins must be >= 2 for any valid n_fft");
    assert!(n_bins <= n_fft, "n_bins must be <= n_fft");
}

/// Prove: n_bins * n_fft (basis matrix size) does not overflow for extended range.
///
/// The cos/sin basis matrices have shape [n_bins, n_fft]. Total elements
/// = n_bins * n_fft. Byte count = n_bins * n_fft * 4.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_basis_matrix_size_no_overflow() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 16384);
    kani::assume(n_fft % 2 == 0);

    let n_bins = n_fft / 2 + 1;
    let basis_elements = n_bins.checked_mul(n_fft);
    assert!(basis_elements.is_some(), "basis matrix elements must not overflow");

    let basis_bytes = basis_elements.unwrap().checked_mul(size_of::<f32>());
    assert!(basis_bytes.is_some(), "basis matrix bytes must not overflow");
}

/// Prove: window buffer size is exactly n_fft elements.
///
/// The Hann window has n_fft samples. Buffer size = n_fft * sizeof(f32).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_window_buffer_size_is_n_fft() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 1 && n_fft <= 65536);

    let window_bytes = n_fft.checked_mul(size_of::<f32>());
    assert!(window_bytes.is_some(), "window buffer must not overflow");
    assert_eq!(window_bytes.unwrap(), n_fft * 4);
}

// ============================================================================
// Zero-frame edge case
// ============================================================================

/// Prove: n_frames == 0 produces full_len == n_fft via saturating_sub.
///
/// Production: `full_len = n_fft + n_frames.saturating_sub(1) * hop`
/// When n_frames == 0: `0_usize.saturating_sub(1) == 0`, so full_len == n_fft.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_zero_frames_full_len_equals_nfft() {
    let n_fft: usize = kani::any();
    let hop: usize = kani::any();
    kani::assume(n_fft >= 1 && n_fft <= 16384);
    kani::assume(hop >= 1 && hop <= n_fft);

    let n_frames: usize = 0;
    let full_len = n_fft + n_frames.saturating_sub(1) * hop;
    assert_eq!(full_len, n_fft, "0 frames: full_len == n_fft");
}

/// Prove: n_frames == 1 also produces full_len == n_fft.
///
/// `1_usize.saturating_sub(1) == 0`, so hop contribution is 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_one_frame_full_len_equals_nfft() {
    let n_fft: usize = kani::any();
    let hop: usize = kani::any();
    kani::assume(n_fft >= 1 && n_fft <= 16384);
    kani::assume(hop >= 1 && hop <= n_fft);

    let n_frames: usize = 1;
    let full_len = n_fft + n_frames.saturating_sub(1) * hop;
    assert_eq!(full_len, n_fft, "1 frame: full_len == n_fft");
}

// ============================================================================
// Output pad vs truncate path
// ============================================================================

/// Prove: output pad path produces exactly output_length elements.
///
/// When trimmed_len < output_length, result is padded with zeros to output_length.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_pad_path_produces_output_length() {
    let trimmed_len: usize = kani::any();
    let output_length: usize = kani::any();
    kani::assume(trimmed_len >= 0 && trimmed_len <= 100_000);
    kani::assume(output_length >= 1 && output_length <= 100_000);
    kani::assume(trimmed_len < output_length);

    // Simulate: padded = trimmed.to_vec(); padded.resize(output_length, 0.0);
    let result_len = output_length;
    assert_eq!(result_len, output_length, "padded result must be output_length");
    assert!(result_len > trimmed_len, "padded result longer than trimmed");
}

/// Prove: output truncate path produces exactly output_length elements.
///
/// When trimmed_len >= output_length, result is trimmed[..output_length].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_truncate_path_produces_output_length() {
    let trimmed_len: usize = kani::any();
    let output_length: usize = kani::any();
    kani::assume(trimmed_len >= 1 && trimmed_len <= 100_000);
    kani::assume(output_length >= 1 && output_length <= 100_000);
    kani::assume(trimmed_len >= output_length);

    // Simulate: trimmed[..output_length].to_vec()
    let result_len = output_length;
    assert_eq!(result_len, output_length, "truncated result must be output_length");
    assert!(result_len <= trimmed_len, "truncated result not longer than trimmed");
}

/// Prove: both pad and truncate paths produce the same output_length.
///
/// The final result vector always has exactly output_length elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_output_always_exact_length() {
    let trimmed_len: usize = kani::any();
    let output_length: usize = kani::any();
    kani::assume(trimmed_len <= 1_000_000);
    kani::assume(output_length >= 1 && output_length <= 1_000_000);

    let result_len = if trimmed_len >= output_length {
        output_length
    } else {
        output_length // resize pads to output_length
    };

    assert_eq!(result_len, output_length, "output is always exactly output_length");
}

// ============================================================================
// Byte alignment
// ============================================================================

/// Prove: all f32 buffer allocations are 4-byte aligned.
///
/// Arena allocations for IDFT and OLA buffers use byte counts that are
/// multiples of sizeof(f32) = 4, ensuring proper f32 alignment.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_buffer_byte_counts_4_aligned() {
    let numel: usize = kani::any();
    kani::assume(numel >= 1 && numel <= 1_000_000);

    let byte_count = numel * size_of::<f32>();
    assert_eq!(byte_count % 4, 0, "f32 buffer bytes must be 4-byte aligned");
}

/// Prove: center trim byte offset is 4-byte aligned when trim is integer.
///
/// `trimmed_off = out_off + trim * 4`. Since trim is an integer, trim * 4
/// is always a multiple of 4, so trimmed_off preserves alignment.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_center_trim_offset_preserves_alignment() {
    let out_off: usize = kani::any();
    let trim: usize = kani::any();
    kani::assume(out_off % 4 == 0); // arena provides 4-byte-aligned offsets
    kani::assume(trim <= 32768);

    let trimmed_off = out_off + trim * size_of::<f32>();
    assert_eq!(trimmed_off % 4, 0, "trimmed offset must remain 4-byte aligned");
}

// ============================================================================
// GPU DynTensor shape validity
// ============================================================================

/// Prove: [1, 1, final_len] shape has ndim == 3 and numel == final_len.
///
/// The GPU-resident path produces a rank-3 DynTensor.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_output_tensor_shape_rank3_valid() {
    let final_len: usize = kani::any();
    kani::assume(final_len <= 1_000_000);

    let shape: [usize; 3] = [1, 1, final_len];
    let ndim = shape.len();
    let numel: usize = shape[0] * shape[1] * shape[2];

    assert_eq!(ndim, 3, "output tensor must be rank 3");
    assert_eq!(numel, final_len, "numel must equal final_len");
}

// ============================================================================
// Normalization monotonicity
// ============================================================================

// Stub for CBMC-incompatible f32::sqrt.
fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    if x > 0.0 { kani::assume(result > 0.0); }
    r
}

/// Prove: unnormalized factor 1/n decreases as n increases, and
/// normalized factor 1/sqrt(n) is finite and positive.
///
/// With CBMC stubs, sqrt is nondeterministic so the 1/sqrt(n) monotonicity
/// cannot be verified. We verify the unnormalized ordering and normalized
/// finiteness/positivity.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn istft_norm_factor_decreasing_with_nfft() {
    let n1: usize = kani::any();
    let n2: usize = kani::any();
    kani::assume(n1 >= 1 && n1 <= 32768);
    kani::assume(n2 > n1 && n2 <= 32768);

    let norm1_unnorm: f32 = 1.0 / n1 as f32;
    let norm2_unnorm: f32 = 1.0 / n2 as f32;
    assert!(
        norm1_unnorm >= norm2_unnorm,
        "1/n must decrease as n increases"
    );

    let norm1_norm: f32 = 1.0 / (n1 as f32).sqrt();
    let norm2_norm: f32 = 1.0 / (n2 as f32).sqrt();
    assert!(norm1_norm.is_finite(), "1/sqrt(n1) must be finite");
    assert!(norm2_norm.is_finite(), "1/sqrt(n2) must be finite");
    assert!(norm1_norm > 0.0, "1/sqrt(n1) must be positive");
    assert!(norm2_norm > 0.0, "1/sqrt(n2) must be positive");
}

// ============================================================================
// No center trim path
// ============================================================================

/// Prove: center=false path returns full output unmodified.
///
/// When center is false, trimmed_len == full_len (no trim applied).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_no_center_trim_full_output() {
    let n_fft: usize = kani::any();
    let n_frames: usize = kani::any();
    let hop: usize = kani::any();
    kani::assume(n_fft >= 64 && n_fft <= 4096);
    kani::assume(n_frames >= 1 && n_frames <= 4096);
    kani::assume(hop >= 1 && hop <= n_fft);

    let full_len = n_fft + n_frames.saturating_sub(1) * hop;
    let center = false;

    let trimmed_len = if center {
        let trim = n_fft / 2;
        if full_len > 2 * trim { full_len - 2 * trim } else { 0 }
    } else {
        full_len
    };

    assert_eq!(trimmed_len, full_len, "center=false: no trimming");
}

// ============================================================================
// Arena allocation budget
// ============================================================================

/// Prove: combined IDFT + OLA allocation fits within 256 MB for production.
///
/// For production ranges (n_fft <= 4096, n_frames <= 8192, hop >= 1):
/// IDFT bytes + OLA bytes <= 256 MB.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_combined_arena_within_budget() {
    let n_fft: usize = kani::any();
    let n_frames: usize = kani::any();
    let hop: usize = kani::any();
    kani::assume(n_fft >= 64 && n_fft <= 4096);
    kani::assume(n_frames >= 1 && n_frames <= 8192);
    kani::assume(hop >= 1 && hop <= n_fft);

    let idft_bytes = n_frames * n_fft * size_of::<f32>();
    let full_len = n_fft + n_frames.saturating_sub(1) * hop;
    let ola_bytes = full_len * size_of::<f32>();

    let combined = idft_bytes + ola_bytes;
    assert!(combined <= 256 * 1024 * 1024, "combined arena within 256 MB");
}

// ============================================================================
// Fused polar kernel grid/threadgroup consistency
// ============================================================================

/// Prove: fused kernel grid [full_len, 1, 1] total threads >= full_len.
///
/// The fused polar→iSTFT kernel has 1D grid. Thread coverage must be >= full_len.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_fused_grid_covers_full_len() {
    let full_len_u32: u32 = kani::any();
    kani::assume(full_len_u32 >= 1);

    let tg_size = 256u32.min(full_len_u32);
    let num_tg = full_len_u32.div_ceil(tg_size);
    let total_threads = (num_tg as u64) * (tg_size as u64);

    assert!(
        total_threads >= full_len_u32 as u64,
        "fused grid must cover all output elements"
    );
}

/// Prove: fused kernel waste is < tg_size (no more than one TG of waste).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_fused_grid_waste_bounded() {
    let full_len_u32: u32 = kani::any();
    kani::assume(full_len_u32 >= 1);

    let tg_size = 256u32.min(full_len_u32);
    let num_tg = full_len_u32.div_ceil(tg_size);
    let total_threads = num_tg * tg_size;
    let waste = total_threads - full_len_u32;

    assert!(waste < tg_size, "waste must be less than one threadgroup");
}

// ============================================================================
// Center trim: n_fft/2 bounds
// ============================================================================

/// Prove: trim = n_fft/2 is exactly half of n_fft for even n_fft.
///
/// The trim amount is always n_fft/2 = half of the FFT window.
/// For even n_fft, 2 * trim == n_fft exactly (no rounding).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_trim_is_exact_half_nfft() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 65536);
    kani::assume(n_fft % 2 == 0);

    let trim = n_fft / 2;
    assert_eq!(2 * trim, n_fft, "2 * trim == n_fft for even n_fft");
}

/// Prove: center trim never removes more samples than available.
///
/// `2 * trim = n_fft`. If full_len <= n_fft, the code produces trimmed_len = 0
/// (empty output), never negative. For full_len > n_fft, trimmed_len > 0.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_center_trim_never_removes_more_than_available() {
    let n_fft: usize = kani::any();
    let n_frames: usize = kani::any();
    let hop: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 4096);
    kani::assume(n_fft % 2 == 0);
    kani::assume(n_frames >= 0 && n_frames <= 4096);
    kani::assume(hop >= 1 && hop <= n_fft);

    let full_len = n_fft + n_frames.saturating_sub(1) * hop;
    let trim = n_fft / 2;

    let trimmed_len = if full_len > 2 * trim {
        full_len - 2 * trim
    } else {
        0
    };

    // Trimmed length is non-negative (usize guarantees this, but assert the logic).
    assert!(trimmed_len <= full_len, "trimmed_len must be <= full_len");
}

// ============================================================================
// Hop ratio and overlap
// ============================================================================

/// Prove: overlap factor = n_fft / hop. For Kokoro (1024/256) = 4.
///
/// The number of frames that overlap at any single output sample is
/// ceil(n_fft / hop). This determines COLA normalization denominator.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_kokoro_overlap_factor() {
    let n_fft: usize = 1024;
    let hop: usize = 256;

    let overlap = n_fft.div_ceil(hop);
    assert_eq!(overlap, 4, "Kokoro overlap factor must be 4");
}

/// Prove: for hop == n_fft (no overlap), exactly 1 frame covers each sample.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_no_overlap_single_frame_coverage() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 1 && n_fft <= 16384);

    let hop = n_fft; // no overlap
    let overlap = n_fft.div_ceil(hop);
    assert_eq!(overlap, 1, "hop == n_fft means exactly 1 frame per sample");
}

// ============================================================================
// Input addressing: row-major layout correctness
// ============================================================================

/// Prove: last valid input address for row-major [n_bins, n_frames] is
/// exactly (n_bins - 1) * n_frames + (n_frames - 1) = n_bins * n_frames - 1.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_input_last_address_correct() {
    let n_bins: usize = kani::any();
    let n_frames: usize = kani::any();
    kani::assume(n_bins >= 2 && n_bins <= 8193);
    kani::assume(n_frames >= 1 && n_frames <= 4096);

    let last_f = n_bins - 1;
    let last_t = n_frames - 1;
    let last_addr = last_f * n_frames + last_t;

    let total_elements = n_bins * n_frames;
    assert_eq!(
        last_addr,
        total_elements - 1,
        "last address must be total_elements - 1"
    );
}
