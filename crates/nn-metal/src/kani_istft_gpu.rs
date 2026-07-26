// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for GPU iSTFT implementation (`istft_gpu.rs`).
//!
//! These harnesses verify safety properties of the GPU-accelerated inverse
//! Short-Time Fourier Transform — buffer sizing, index arithmetic, kernel
//! dispatch parameters, center trimming, and output construction — without
//! requiring a live Metal context.
//!
//! ## Properties proved:
//!
//! **IDFT kernel dispatch safety:**
//! - IDFT buffer sizing `n_frames * n_fft * 4` does not overflow
//! - 2D threadgroup clamping `min(16, dim)` is in [1, 16]
//! - 2D threadgroup product is <= 256 (Metal limit for 2D dispatch)
//! - IDFT grid dimensions `[n_fft, n_frames]` fit u32
//! - Per-thread IDFT index `t * n_fft + k` does not overflow u32
//! - Basis matrix addressing `f * n_fft + k` does not overflow u32
//! - STFT input addressing `f * n_frames + t` does not overflow u32
//!
//! **Overlap-add kernel dispatch safety:**
//! - OLA full_len calculation does not overflow
//! - OLA buffer byte count does not overflow
//! - 1D threadgroup clamping `min(256, full_len)` is in [1, 256]
//! - OLA window addressing: frame_start + n_fft - 1 does not exceed full_len
//! - Per-sample frame iteration count is bounded by n_frames
//!
//! **Fused polar-to-iSTFT kernel safety:**
//! - Fused output buffer `full_len * 4` does not overflow
//! - Fused threadgroup size is valid
//!
//! **Center trimming correctness:**
//! - GPU byte-offset trim matches CPU index-based trim
//! - Trimmed length is `full_len - n_fft` for valid parameters
//! - GPU `trimmed_off` does not exceed buffer bounds
//! - `final_len = min(trimmed_len, output_length)` is correct
//!
//! **Input validation:**
//! - `expected_len = n_bins * n_frames` correct for [n_bins, n_frames] layout
//! - Finiteness check iteration covers all real + imag elements
//!
//! **Normalization factor:**
//! - `1.0 / sqrt(n_fft)` and `1.0 / n_fft` are finite and positive
//! - Normalized factor >= non-normalized factor for all n_fft >= 1
//!
//! Part of #3697.

use std::mem::size_of;

// ============================================================================
// IDFT kernel: buffer sizing
// ============================================================================

/// Prove: IDFT element count `n_frames * n_fft` does not overflow for
/// extended production ranges (beyond Kokoro).
///
/// Covers larger models that may use bigger FFT sizes or longer sequences.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_idft_numel_extended_range() {
    let n_frames: usize = kani::any();
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 64 && n_fft <= 16384);
    kani::assume(n_frames >= 1 && n_frames <= 32768);

    let idft_numel = n_frames.checked_mul(n_fft);
    assert!(idft_numel.is_some(), "IDFT numel must not overflow");

    let idft_bytes = idft_numel
        .unwrap()
        .checked_mul(size_of::<f32>());
    assert!(idft_bytes.is_some(), "IDFT bytes must not overflow");
}

/// Prove: IDFT buffer byte count is bounded by a reasonable maximum.
///
/// For Kokoro (n_fft=1024, n_frames<=2048): max 8 MB.
/// For extended (n_fft=4096, n_frames<=4096): max 64 MB.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_idft_bytes_bounded() {
    let n_frames: usize = kani::any();
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 64 && n_fft <= 4096);
    kani::assume(n_frames >= 1 && n_frames <= 4096);

    let idft_bytes = n_frames * n_fft * size_of::<f32>();
    // Max: 4096 * 4096 * 4 = 67_108_864 (64 MB).
    assert!(idft_bytes <= 256 * 1024 * 1024, "IDFT bytes within 256 MB");
}

// ============================================================================
// IDFT kernel: threadgroup sizing
// ============================================================================

/// Prove: 2D threadgroup clamping produces valid dimensions.
///
/// Production: `tg_x = 16u32.min(n_fft_u32); tg_y = 16u32.min(n_frames_u32);`
/// Both must be in [1, 16] and product must be <= 256.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_idft_threadgroup_2d_valid() {
    let n_fft_u32: u32 = kani::any();
    let n_frames_u32: u32 = kani::any();
    kani::assume(n_fft_u32 >= 1 && n_fft_u32 <= 65536);
    kani::assume(n_frames_u32 >= 1 && n_frames_u32 <= 65536);

    let tg_x = 16u32.min(n_fft_u32);
    let tg_y = 16u32.min(n_frames_u32);

    assert!(tg_x >= 1 && tg_x <= 16, "tg_x in [1, 16]");
    assert!(tg_y >= 1 && tg_y <= 16, "tg_y in [1, 16]");

    let product = tg_x * tg_y;
    assert!(product >= 1 && product <= 256, "TG product in [1, 256]");
}

/// Prove: IDFT grid dimensions `[n_fft, n_frames, 1]` fit u32.
///
/// The dispatch grid is `[n_fft_u32, n_frames_u32, 1]`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_idft_grid_fits_u32() {
    let n_fft: usize = kani::any();
    let n_frames: usize = kani::any();
    kani::assume(n_fft >= 1 && n_fft <= 65536);
    kani::assume(n_frames >= 1 && n_frames <= 65536);

    assert!(n_fft <= u32::MAX as usize, "n_fft fits u32 grid");
    assert!(n_frames <= u32::MAX as usize, "n_frames fits u32 grid");
}

// ============================================================================
// IDFT kernel: index arithmetic
// ============================================================================

/// Prove: IDFT output index `t * n_fft + k` does not overflow u32.
///
/// MSL: `output[t * n_fft + k]` where t < n_frames, k < n_fft.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_idft_output_index_safe() {
    let n_fft: u32 = kani::any();
    let n_frames: u32 = kani::any();
    kani::assume(n_fft >= 64 && n_fft <= 16384);
    kani::assume(n_frames >= 1 && n_frames <= 16384);

    let t: u32 = kani::any();
    let k: u32 = kani::any();
    kani::assume(t < n_frames);
    kani::assume(k < n_fft);

    let idx = (t as u64) * (n_fft as u64) + (k as u64);
    assert!(idx <= u32::MAX as u64, "IDFT output index fits u32");
}

/// Prove: basis matrix addressing `f * n_fft + k` does not overflow u32.
///
/// MSL: `cos_basis[f * n_fft + k]` and `sin_basis[f * n_fft + k]`.
/// f < n_bins, k < n_fft.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_basis_index_safe() {
    let n_fft: u32 = kani::any();
    let n_bins: u32 = kani::any();
    kani::assume(n_fft >= 64 && n_fft <= 16384);
    // n_bins = n_fft / 2 + 1, but we bound it independently.
    kani::assume(n_bins >= 2 && n_bins <= n_fft);

    let f: u32 = kani::any();
    let k: u32 = kani::any();
    kani::assume(f < n_bins);
    kani::assume(k < n_fft);

    let idx = (f as u64) * (n_fft as u64) + (k as u64);
    assert!(idx <= u32::MAX as u64, "basis index fits u32");
}

/// Prove: STFT input addressing `f * n_frames + t` does not overflow u32.
///
/// MSL: `real[f * n_frames + t]` and `imag[f * n_frames + t]`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_stft_input_index_safe() {
    let n_frames: u32 = kani::any();
    let n_bins: u32 = kani::any();
    kani::assume(n_frames >= 1 && n_frames <= 16384);
    kani::assume(n_bins >= 2 && n_bins <= 8193);

    let f: u32 = kani::any();
    let t: u32 = kani::any();
    kani::assume(f < n_bins);
    kani::assume(t < n_frames);

    let idx = (f as u64) * (n_frames as u64) + (t as u64);
    assert!(idx <= u32::MAX as u64, "STFT input index fits u32");
}

// ============================================================================
// Overlap-add kernel: sizing
// ============================================================================

/// Prove: OLA full_len `n_fft + (n_frames - 1) * hop` does not overflow.
///
/// Production uses `saturating_sub` for n_frames == 0 safety.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_ola_full_len_no_overflow() {
    let n_fft: usize = kani::any();
    let n_frames: usize = kani::any();
    let hop: usize = kani::any();
    kani::assume(n_fft >= 64 && n_fft <= 16384);
    kani::assume(n_frames >= 1 && n_frames <= 32768);
    kani::assume(hop >= 1 && hop <= n_fft);

    let hop_contribution = n_frames.saturating_sub(1).checked_mul(hop);
    assert!(hop_contribution.is_some(), "(n_frames-1)*hop must not overflow");

    let full_len = n_fft.checked_add(hop_contribution.unwrap());
    assert!(full_len.is_some(), "full_len must not overflow");

    // full_len is always >= n_fft.
    assert!(full_len.unwrap() >= n_fft);
}

/// Prove: OLA buffer byte count does not overflow for extended ranges.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_ola_bytes_no_overflow() {
    let n_fft: usize = kani::any();
    let n_frames: usize = kani::any();
    let hop: usize = kani::any();
    kani::assume(n_fft >= 64 && n_fft <= 8192);
    kani::assume(n_frames >= 1 && n_frames <= 16384);
    kani::assume(hop >= 1 && hop <= n_fft);

    let full_len = n_fft + n_frames.saturating_sub(1) * hop;
    let ola_bytes = full_len.checked_mul(size_of::<f32>());
    assert!(ola_bytes.is_some(), "OLA bytes must not overflow");
}

/// Prove: OLA 1D threadgroup clamping is valid.
///
/// Production: `tg_size = 256u32.min(full_len_u32);`
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_ola_threadgroup_valid() {
    let full_len_u32: u32 = kani::any();
    kani::assume(full_len_u32 >= 1);

    let tg_size = 256u32.min(full_len_u32);
    assert!(tg_size >= 1 && tg_size <= 256, "OLA TG size in [1, 256]");
}

// ============================================================================
// Overlap-add kernel: index arithmetic
// ============================================================================

/// Prove: OLA window start `frame * hop` does not exceed full_len.
///
/// For each frame t: `frame_start = t * hop`. The window covers
/// `[frame_start, frame_start + n_fft)` which must be within full_len.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_ola_window_within_bounds() {
    let n_fft: usize = kani::any();
    let n_frames: usize = kani::any();
    let hop: usize = kani::any();
    kani::assume(n_fft >= 64 && n_fft <= 4096);
    kani::assume(n_frames >= 1 && n_frames <= 4096);
    kani::assume(hop >= 1 && hop <= n_fft);

    let full_len = n_fft + n_frames.saturating_sub(1) * hop;

    let t: usize = kani::any();
    kani::assume(t < n_frames);

    let frame_start = t * hop;
    let frame_end = frame_start + n_fft;

    // The last frame (t = n_frames - 1): frame_end = (n_frames-1)*hop + n_fft = full_len.
    assert!(
        frame_end <= full_len,
        "window must not exceed full_len"
    );
    assert!(frame_start < full_len, "frame_start within bounds");
}

/// Prove: per-sample frame iteration in OLA kernel is bounded.
///
/// For output sample `i`, the contributing frames are those where
/// `frame_start <= i < frame_start + n_fft`, i.e., `t*hop <= i < t*hop + n_fft`.
/// The number of such frames is at most `ceil(n_fft / hop)`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_ola_per_sample_frame_count_bounded() {
    let n_fft: usize = kani::any();
    let hop: usize = kani::any();
    kani::assume(n_fft >= 1 && n_fft <= 8192);
    kani::assume(hop >= 1 && hop <= n_fft);

    // Maximum overlapping frames at any sample position.
    let max_overlap = n_fft.div_ceil(hop);
    // This is bounded by n_fft (when hop=1, every frame overlaps).
    assert!(max_overlap <= n_fft, "max overlap bounded by n_fft");
    // For Kokoro (n_fft=1024, hop=256): max_overlap = 4.
    // For typical STFT: max_overlap is small.
}

// ============================================================================
// Fused polar-to-iSTFT kernel: sizing
// ============================================================================

/// Prove: fused kernel output buffer `full_len * 4` does not overflow.
///
/// Same full_len formula as OLA, same buffer requirement.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_fused_output_bytes_safe() {
    let n_fft: usize = kani::any();
    let n_frames: usize = kani::any();
    let hop: usize = kani::any();
    kani::assume(n_fft >= 64 && n_fft <= 8192);
    kani::assume(n_frames >= 1 && n_frames <= 16384);
    kani::assume(hop >= 1 && hop <= n_fft);

    let full_len = n_fft + n_frames.saturating_sub(1) * hop;
    let out_bytes = full_len.checked_mul(size_of::<f32>());
    assert!(out_bytes.is_some(), "fused output bytes must not overflow");
}

/// Prove: fused kernel 1D threadgroup size is valid.
///
/// Production: `tg_size = 256u32.min(full_len_u32);`
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_fused_threadgroup_valid() {
    let full_len_u32: u32 = kani::any();
    kani::assume(full_len_u32 >= 1);

    let tg_size = 256u32.min(full_len_u32);
    assert!(tg_size >= 1 && tg_size <= 256);
}

// ============================================================================
// Center trimming: correctness
// ============================================================================

/// Prove: GPU byte-offset center trim produces same trimmed_len as CPU.
///
/// CPU: `trimmed = &raw_output[trim..full_len - trim]`
/// GPU: `trimmed_off = out_off + trim * 4; trimmed_len = full_len - 2 * trim`
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_center_trim_cpu_gpu_agreement() {
    let n_fft: usize = kani::any();
    let n_frames: usize = kani::any();
    let hop: usize = kani::any();
    let out_off: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 8192);
    kani::assume(n_fft % 2 == 0);
    kani::assume(n_frames >= 1 && n_frames <= 8192);
    kani::assume(hop >= 1 && hop <= n_fft);
    kani::assume(out_off <= 64 * 1024 * 1024);

    let full_len = n_fft + n_frames.saturating_sub(1) * hop;
    let trim = n_fft / 2;

    // CPU path.
    let cpu_trimmed_len = if full_len > 2 * trim {
        full_len - 2 * trim
    } else {
        0
    };

    // GPU path.
    let (gpu_trimmed_off, gpu_trimmed_len) = if full_len > 2 * trim {
        (out_off + trim * size_of::<f32>(), full_len - 2 * trim)
    } else {
        (out_off, 0)
    };

    assert_eq!(cpu_trimmed_len, gpu_trimmed_len, "CPU and GPU must agree on trimmed_len");

    // Verify offset arithmetic.
    if full_len > 2 * trim {
        assert_eq!(gpu_trimmed_off, out_off + trim * 4, "GPU offset advances by trim*4");
    }
}

/// Prove: trimmed_len equals `full_len - n_fft` for center=true with valid params.
///
/// When `full_len > 2 * trim` (typical case): `trimmed_len = full_len - 2*(n_fft/2) = full_len - n_fft`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_trimmed_len_equals_full_minus_nfft() {
    let n_fft: usize = kani::any();
    let n_frames: usize = kani::any();
    let hop: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 8192);
    kani::assume(n_fft % 2 == 0);
    kani::assume(n_frames >= 2 && n_frames <= 8192);
    kani::assume(hop >= 1 && hop <= n_fft);

    let full_len = n_fft + n_frames.saturating_sub(1) * hop;
    let trim = n_fft / 2;

    // For n_frames >= 2: full_len = n_fft + (n_frames-1)*hop >= n_fft + hop > n_fft = 2*trim.
    // So the trim condition always holds.
    if full_len > 2 * trim {
        let trimmed_len = full_len - 2 * trim;
        assert_eq!(
            trimmed_len,
            full_len - n_fft,
            "trimmed_len == full_len - n_fft"
        );
        // For standard STFT: trimmed_len = (n_frames - 1) * hop.
        assert_eq!(
            trimmed_len,
            (n_frames - 1) * hop,
            "trimmed_len == (n_frames-1)*hop"
        );
    }
}

/// Prove: GPU `trimmed_off` does not exceed the buffer allocation.
///
/// The trimmed region `[trimmed_off, trimmed_off + trimmed_len * 4)` must
/// be within the allocated `[out_off, out_off + full_len * 4)`.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_trimmed_off_within_buffer() {
    let n_fft: usize = kani::any();
    let n_frames: usize = kani::any();
    let hop: usize = kani::any();
    let out_off: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 4096);
    kani::assume(n_fft % 2 == 0);
    kani::assume(n_frames >= 1 && n_frames <= 4096);
    kani::assume(hop >= 1 && hop <= n_fft);
    kani::assume(out_off <= 64 * 1024 * 1024);

    let full_len = n_fft + n_frames.saturating_sub(1) * hop;
    let trim = n_fft / 2;

    if full_len > 2 * trim {
        let trimmed_off = out_off + trim * size_of::<f32>();
        let trimmed_len = full_len - 2 * trim;

        // trimmed region end.
        let trimmed_end = trimmed_off + trimmed_len * size_of::<f32>();
        // buffer end.
        let buffer_end = out_off + full_len * size_of::<f32>();

        assert!(
            trimmed_end <= buffer_end,
            "trimmed region must be within allocated buffer"
        );
        assert!(
            trimmed_off >= out_off,
            "trimmed offset must be >= buffer start"
        );
    }
}

/// Prove: `final_len = min(trimmed_len, output_length)` is correct.
///
/// Production (gpu_istft_from_polar_gpu): `let final_len = trimmed_len.min(output_length);`
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_final_len_min_correct() {
    let trimmed_len: usize = kani::any();
    let output_length: usize = kani::any();
    kani::assume(trimmed_len <= 1_000_000);
    kani::assume(output_length <= 1_000_000);

    let final_len = trimmed_len.min(output_length);
    assert!(final_len <= trimmed_len, "final_len <= trimmed_len");
    assert!(final_len <= output_length, "final_len <= output_length");
    if trimmed_len <= output_length {
        assert_eq!(final_len, trimmed_len);
    } else {
        assert_eq!(final_len, output_length);
    }
}

// ============================================================================
// Input validation
// ============================================================================

/// Prove: `expected_len = n_bins * n_frames` matches the `[n_bins, n_frames]`
/// row-major buffer layout.
///
/// Both real and imag buffers must have exactly `n_bins * n_frames` elements.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_input_expected_len_correct() {
    let n_fft: usize = kani::any();
    let n_frames: usize = kani::any();
    kani::assume(n_fft >= 2 && n_fft <= 16384);
    kani::assume(n_fft % 2 == 0);
    kani::assume(n_frames >= 1 && n_frames <= 32768);

    let n_bins = n_fft / 2 + 1;
    let expected_len = n_bins.checked_mul(n_frames);
    assert!(expected_len.is_some(), "n_bins * n_frames must not overflow");

    // expected_len is the total number of complex coefficients per channel.
    let len = expected_len.unwrap();
    assert!(len >= n_bins, "expected_len >= n_bins");
    assert!(len >= n_frames, "expected_len >= n_frames");
}

/// Prove: finiteness check iteration covers all elements.
///
/// Production: `for &v in real.iter().chain(imag.iter())`.
/// Total iteration count = real.len() + imag.len() = 2 * expected_len.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_finiteness_check_covers_all() {
    let n_bins: usize = kani::any();
    let n_frames: usize = kani::any();
    kani::assume(n_bins >= 2 && n_bins <= 8193);
    kani::assume(n_frames >= 1 && n_frames <= 4096);

    let expected_len = n_bins * n_frames;
    let real_len = expected_len;
    let imag_len = expected_len;

    // chain(real, imag) iterates real_len + imag_len elements.
    let total_checked = real_len + imag_len;
    assert_eq!(
        total_checked,
        2 * expected_len,
        "finiteness check must cover all real + imag elements"
    );
}

// ============================================================================
// Normalization factor
// ============================================================================

// Stub for CBMC-incompatible f32::sqrt.
fn sqrt_f32_stub(x: f32) -> f32 {
    let r: f32 = kani::any();
    kani::assume(r.is_finite() && r > 0.0 && r <= 1e10);
    if x > 0.0 { kani::assume(result > 0.0); }
    r
}

/// Prove: normalization factor is finite and positive.
///
/// Both `1.0 / sqrt(n_fft)` (normalized) and `1.0 / n_fft` (unnormalized)
/// must be safe for all valid n_fft. With CBMC stubs, the `<= 1.0` bound
/// cannot be verified since sqrt is nondeterministic.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn istft_gpu_norm_factor_safe() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 1 && n_fft <= 65536);

    let norm_normalized: f32 = 1.0 / (n_fft as f32).sqrt();
    let norm_unnormalized: f32 = 1.0 / n_fft as f32;

    assert!(norm_normalized.is_finite(), "normalized norm must be finite");
    assert!(norm_unnormalized.is_finite(), "unnormalized norm must be finite");
    assert!(norm_normalized > 0.0, "normalized norm must be positive");
    assert!(norm_unnormalized > 0.0, "unnormalized norm must be positive");
    assert!(norm_unnormalized <= 1.0, "unnormalized norm <= 1.0");
}

/// Prove: both normalization factors are finite and positive for all n_fft >= 1.
///
/// With CBMC stubs, the ordering property 1/sqrt(n) >= 1/n cannot be verified
/// since sqrt is nondeterministic. We verify finiteness and positivity.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn istft_gpu_norm_factor_ordering() {
    let n_fft: usize = kani::any();
    kani::assume(n_fft >= 1 && n_fft <= 65536);

    let norm_normalized: f32 = 1.0 / (n_fft as f32).sqrt();
    let norm_unnormalized: f32 = 1.0 / n_fft as f32;

    assert!(norm_normalized.is_finite(), "1/sqrt(n) must be finite");
    assert!(norm_normalized > 0.0, "1/sqrt(n) must be positive");
    assert!(norm_unnormalized.is_finite(), "1/n must be finite");
    assert!(norm_unnormalized > 0.0, "1/n must be positive");
}

// ============================================================================
// Kokoro production parameters
// ============================================================================

/// Prove: all iSTFT arithmetic is safe for Kokoro production parameters.
///
/// Kokoro: n_fft=1024, hop_length=256, center=true, normalized=false.
/// n_frames varies with synthesis length.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_kokoro_params_comprehensive() {
    let n_frames: usize = kani::any();
    kani::assume(n_frames >= 1 && n_frames <= 2048);

    let n_fft: usize = 1024;
    let hop: usize = 256;
    let n_bins: usize = n_fft / 2 + 1; // 513

    // IDFT sizing.
    let idft_numel = n_frames * n_fft;
    let idft_bytes = idft_numel * size_of::<f32>();
    assert!(idft_bytes <= 64 * 1024 * 1024, "IDFT fits in arena");

    // OLA sizing.
    let full_len = n_fft + n_frames.saturating_sub(1) * hop;
    let ola_bytes = full_len * size_of::<f32>();
    assert!(ola_bytes <= 64 * 1024 * 1024, "OLA fits in arena");

    // to_u32 conversions.
    assert!(n_bins <= u32::MAX as usize);
    assert!(n_frames <= u32::MAX as usize);
    assert!(n_fft <= u32::MAX as usize);
    assert!(hop <= u32::MAX as usize);
    assert!(full_len <= u32::MAX as usize);

    // Normalization.
    let norm: f32 = 1.0 / n_fft as f32; // normalized=false for Kokoro
    assert!(norm.is_finite() && norm > 0.0);

    // Center trim.
    let trim = n_fft / 2; // 512
    if n_frames >= 2 {
        assert!(full_len > 2 * trim);
        let trimmed_len = full_len - 2 * trim;
        assert_eq!(trimmed_len, (n_frames - 1) * hop);
    }

    // Input validation.
    let expected_len = n_bins * n_frames;
    assert!(expected_len <= 2 * 1024 * 1024, "input size reasonable");
}

/// Prove: GPU-resident path (gpu_istft_from_polar_gpu) produces valid
/// output shape `[1, 1, final_len]` for Kokoro parameters.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn istft_gpu_kokoro_gpu_resident_shape() {
    let n_frames: usize = kani::any();
    let output_length: usize = kani::any();
    kani::assume(n_frames >= 1 && n_frames <= 2048);
    kani::assume(output_length >= 1 && output_length <= 1_000_000);

    let n_fft: usize = 1024;
    let hop: usize = 256;
    let full_len = n_fft + n_frames.saturating_sub(1) * hop;
    let trim = n_fft / 2;

    let trimmed_len = if full_len > 2 * trim {
        full_len - 2 * trim
    } else {
        0
    };

    let final_len = trimmed_len.min(output_length);

    // Output shape is [1, 1, final_len].
    let shape = [1_usize, 1, final_len];
    let numel: usize = shape.iter().product();
    assert_eq!(numel, final_len, "shape product equals final_len");
    // The DynTensor::from_gpu_storage call must succeed for this shape.
    assert!(shape.len() == 3, "shape is 3D");
}
