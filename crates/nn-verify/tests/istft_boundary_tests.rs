// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Boundary condition tests for iSTFT weight matrix builder (#3351 T3.6).
//!
//! Covers edge cases not exercised by the inline unit tests:
//! - Minimum valid n_fft=2 (only DC + Nyquist, no interior frequencies)
//! - center=false (no trimming)
//! - Single frame (n_frames=1, center=false)
//! - Partial output_length < trimmed_len
//!
//! Part of algorithm_audit phase: proving boundary correctness of the
//! iSTFT linear matrix used for CROWN bound propagation.

use nn_verify::istft_linear_matrix::build_istft_weight_matrix;

/// Minimum valid n_fft=2: only DC(f=0) and Nyquist(f=1), no interior frequencies.
/// Conjugate symmetry factor `sym` is always 1.0 (both f=0 and f=n_bins-1=1).
/// This is the smallest valid iSTFT — any off-by-one in the frequency loop
/// or symmetry factor would produce incorrect weights.
#[test]
fn test_istft_boundary_minimum_nfft_2() {
    let n_fft = 2;
    let hop = 1;
    let n_frames = 4;
    // center=true: full_len = 2 + 3*1 = 5, trimmed = 5 - 2 = 3
    let output_length = 3;

    let mat = build_istft_weight_matrix(n_fft, hop, n_frames, output_length, false, true).unwrap();
    let n_bins = n_fft / 2 + 1; // 2: DC(f=0) and Nyquist(f=1)
    assert_eq!(mat.input_dim, 2 * n_bins * n_frames); // 2 * 2 * 4 = 16
    assert_eq!(mat.output_length, 3);

    // All weights must be finite (COLA division must not produce Inf/NaN).
    for (idx, &w) in mat.weights.iter().enumerate() {
        assert!(w.is_finite(), "weight[{idx}] = {w} not finite for n_fft=2");
    }
    let nonzero = mat.weights.iter().filter(|&&w| w.abs() > 1e-10).count();
    assert!(nonzero > 0, "n_fft=2 matrix must have nonzero entries");
}

/// center=false mode (no trimming). Previously untested path.
/// Without center padding, full_len = n_fft + (n_frames-1)*hop and no
/// samples are trimmed. Edge samples have fewer overlapping frames,
/// so COLA window_sum is smaller (but eps guard handles near-zero).
#[test]
fn test_istft_boundary_center_false() {
    let n_fft = 20;
    let hop = 5;
    let n_frames = 10;
    // center=false: full_len = 20 + 9*5 = 65, no trim
    let full_len = n_fft + (n_frames - 1) * hop;
    let output_length = full_len;

    let mat = build_istft_weight_matrix(n_fft, hop, n_frames, output_length, false, false).unwrap();
    assert_eq!(mat.output_length, full_len);
    assert_eq!(mat.input_dim, 2 * (n_fft / 2 + 1) * n_frames);

    for (idx, &w) in mat.weights.iter().enumerate() {
        assert!(
            w.is_finite(),
            "weight[{idx}] = {w} not finite (center=false)"
        );
    }
    let nonzero = mat.weights.iter().filter(|&&w| w.abs() > 1e-10).count();
    assert!(nonzero > 0, "center=false matrix must have nonzero entries");
}

/// Single frame (n_frames=1, center=false). Degenerate overlap-add case:
/// one STFT frame, no overlap. COLA window_sum = Hann(k)^2 for each k.
///
/// Key boundary: Periodic Hann(0) = 0, so the eps guard zeros out row 0.
/// Hann(N-1) != 0 for periodic Hann (divides by N, not N-1).
#[test]
fn test_istft_boundary_single_frame_center_false() {
    let n_fft = 4;
    let hop = 2;
    let n_frames = 1;
    // center=false: full_len = 4 + 0 = 4
    let output_length = n_fft;

    let mat = build_istft_weight_matrix(n_fft, hop, n_frames, output_length, false, false).unwrap();
    let n_bins = n_fft / 2 + 1; // 3
    assert_eq!(mat.input_dim, (2 * n_bins)); // 6
    assert_eq!(mat.output_length, 4);

    for (idx, &w) in mat.weights.iter().enumerate() {
        assert!(
            w.is_finite(),
            "weight[{idx}] = {w} not finite (single frame)"
        );
    }

    // Periodic Hann(0) = 0.5*(1-cos(0)) = 0, so COLA(0) = 0 → row 0 is zero.
    let row0_sum: f32 = (0..mat.input_dim).map(|j| mat.weights[j].abs()).sum();
    assert!(
        row0_sum < 1e-8,
        "row 0 should be zero (Hann(0)=0): sum={row0_sum}"
    );

    // Periodic Hann(1) = 0.5*(1-cos(pi/2)) = 0.5 → nonzero row.
    let row1_sum: f32 = (0..mat.input_dim)
        .map(|j| mat.weights[mat.input_dim + j].abs())
        .sum();
    assert!(
        row1_sum > 1e-6,
        "row 1 should be nonzero (Hann(1) > 0): sum={row1_sum}"
    );

    // Periodic Hann(3) = 0.5*(1-cos(3*pi/2)) = 0.5 → last row is also nonzero.
    let row_last_sum: f32 = (0..mat.input_dim)
        .map(|j| mat.weights[(output_length - 1) * mat.input_dim + j].abs())
        .sum();
    assert!(
        row_last_sum > 1e-6,
        "last row should be nonzero for periodic Hann: sum={row_last_sum}"
    );
}

/// Partial output (output_length < trimmed_len). The first `output_length`
/// rows of the partial matrix must equal the corresponding rows of the
/// full matrix — the builder must not change weights based on output_length.
#[test]
fn test_istft_boundary_partial_output_length() {
    let n_fft = 20;
    let hop = 5;
    let n_frames = 10;
    let full_output = (n_frames - 1) * hop; // 45

    let partial = full_output / 2; // 22
    let mat_partial =
        build_istft_weight_matrix(n_fft, hop, n_frames, partial, false, true).unwrap();
    let mat_full =
        build_istft_weight_matrix(n_fft, hop, n_frames, full_output, false, true).unwrap();

    assert_eq!(mat_partial.output_length, partial);
    assert_eq!(mat_partial.input_dim, mat_full.input_dim);

    // First `partial` rows must match between partial and full matrices.
    for row in 0..partial {
        for col in 0..mat_partial.input_dim {
            let w_partial = mat_partial.weights[row * mat_partial.input_dim + col];
            let w_full = mat_full.weights[row * mat_full.input_dim + col];
            assert!(
                (w_partial - w_full).abs() < 1e-10,
                "row {row} col {col}: partial={w_partial} != full={w_full}"
            );
        }
    }
}

/// output_length = 1 (minimum valid output). Single output sample.
#[test]
fn test_istft_boundary_output_length_one() {
    let n_fft = 4;
    let hop = 1;
    let n_frames = 6;
    let output_length = 1;

    let mat = build_istft_weight_matrix(n_fft, hop, n_frames, output_length, false, true).unwrap();
    assert_eq!(mat.output_length, 1);
    assert_eq!(mat.weights.len(), mat.input_dim);

    for (idx, &w) in mat.weights.iter().enumerate() {
        assert!(w.is_finite(), "weight[{idx}] = {w} not finite (output=1)");
    }
}

/// n_frames=1 with center=true must fail (trimmed_len = 0).
#[test]
fn test_istft_boundary_single_frame_center_true_fails() {
    let n_fft = 4;
    let hop = 1;
    let n_frames = 1;
    // center=true: full_len = 4, trimmed = 4 - 4 = 0
    // Any output_length > 0 exceeds available.
    let result = build_istft_weight_matrix(n_fft, hop, n_frames, 1, false, true);
    assert!(
        result.is_err(),
        "single frame + center=true should fail (trimmed_len=0)"
    );
}
