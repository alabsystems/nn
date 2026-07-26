// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for CPU-side STFT magnitude computation.

use super::*;

// -- StftParams::new() --------------------------------------------------------

#[test]
fn test_stft_params_new_derives_fields() {
    let p = StftParams::new(256, 128);
    assert_eq!(p.n_fft, 256);
    assert_eq!(p.hop_length, 128);
    assert_eq!(p.n_freqs, 129); // 256/2 + 1
    assert_eq!(p.pad_right, 64); // 256/4
}

#[test]
fn test_stft_params_new_small_fft() {
    let p = StftParams::new(4, 2);
    assert_eq!(p.n_freqs, 3); // 4/2 + 1
    assert_eq!(p.pad_right, 1); // 4/4
}

#[test]
fn test_stft_params_new_matches_default() {
    let from_new = StftParams::new(256, 128);
    let from_default = StftParams::default();
    assert_eq!(from_new.n_fft, from_default.n_fft);
    assert_eq!(from_new.hop_length, from_default.hop_length);
    assert_eq!(from_new.n_freqs, from_default.n_freqs);
    assert_eq!(from_new.pad_right, from_default.pad_right);
}

#[test]
fn test_stft_params_new_used_in_compute() {
    // Verify StftParams::new() produces valid params for compute_stft_magnitude.
    let params = StftParams::new(4, 2);
    let audio = vec![1.0f32; 6];
    let basis = vec![0.0f32; 6 * 4]; // (n_fft+2) * n_fft = 24
    let result = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    // n_frames = (6 + 1 - 4) / 2 + 1 = 2 (padded_len = 6 + pad_right=1 = 7)
    assert_eq!(result.len(), 3 * 2); // n_freqs=3, n_frames=2
}

// -- STFT computation ---------------------------------------------------------

#[test]
fn test_stft_silero_vad_output_shape() {
    let params = StftParams::default();
    // 576 samples + 64 pad = 640; (640 - 256) / 128 + 1 = 4 frames
    let audio = vec![0.0f32; 576];
    let basis = vec![0.0f32; 258 * 256];
    let result = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    // [129, 4] = 516 elements
    assert_eq!(result.len(), 129 * 4);
}

#[test]
fn test_stft_basis_size_mismatch() {
    let params = StftParams::default();
    let audio = vec![0.0f32; 576];
    let bad_basis = vec![0.0f32; 100];
    let result = compute_stft_magnitude(&audio, &bad_basis, &params);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(StftError::BasisSizeMismatch { .. })),
        "expected BasisSizeMismatch, got {result:?}"
    );
}

#[test]
fn test_stft_audio_too_short() {
    let params = StftParams {
        n_fft: 256,
        hop_length: 128,
        n_freqs: 129,
        pad_right: 0,
    };
    let audio = vec![0.0f32; 100]; // Too short for n_fft=256
    let basis = vec![0.0f32; 258 * 256];
    let result = compute_stft_magnitude(&audio, &basis, &params);
    assert!(result.is_err());
    assert!(
        matches!(result, Err(StftError::AudioTooShort { .. })),
        "expected AudioTooShort, got {result:?}"
    );
}

#[test]
fn test_stft_known_values() {
    // Simple case: single frequency bin, known dot product
    let params = StftParams {
        n_fft: 4,
        hop_length: 2,
        n_freqs: 3, // n_fft/2 + 1 = 3
        pad_right: 0,
    };
    // Audio: [1, 2, 3, 4, 5, 6]
    // n_frames = (6 - 4) / 2 + 1 = 2
    let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    // Basis: [n_fft+2, 1, n_fft] = [6, 1, 4] = 24 elements
    // Set filter 0 (real freq 0) to [1, 0, 0, 0]
    // Set filter 3 (imag freq 0) to [0, 1, 0, 0]
    let mut basis = vec![0.0f32; 6 * 4];
    basis[0] = 1.0; // filter 0, element 0
    basis[3 * 4 + 1] = 1.0; // filter 3, element 1

    let result = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    assert_eq!(result.len(), 3 * 2); // n_freqs * n_frames

    // Freq 0, frame 0: real = dot([1,0,0,0], [1,2,3,4]) = 1.0
    //                   imag = dot([0,1,0,0], [1,2,3,4]) = 2.0
    //                   mag = sqrt(1 + 4) = sqrt(5)
    let expected_mag = 1.0f32.hypot(2.0);
    assert!(
        (result[0] - expected_mag).abs() < 1e-6,
        "freq=0, t=0: expected {expected_mag}, got {}",
        result[0]
    );

    // Freq 0, frame 1: real = dot([1,0,0,0], [3,4,5,6]) = 3.0
    //                   imag = dot([0,1,0,0], [3,4,5,6]) = 4.0
    //                   mag = sqrt(9 + 16) = 5.0
    assert!(
        (result[1] - 5.0).abs() < 1e-6,
        "freq=0, t=1: expected 5.0, got {}",
        result[1]
    );
}

#[test]
fn test_stft_reflection_padding_correctness() {
    // F3: Use an identity-like basis on a known signal to verify actual
    // reflected sample values, not just frame count.
    let params = StftParams {
        n_fft: 4,
        hop_length: 2,
        n_freqs: 3,
        pad_right: 3,
    };
    // Audio: [10, 20, 30, 40, 50]
    // Reflect pad 3 right: mirror from end excluding boundary
    // audio[3]=40, audio[2]=30, audio[1]=20
    // Padded: [10, 20, 30, 40, 50, 40, 30, 20]
    let audio = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    // Use identity basis for filter 0 (real freq 0): [1, 0, 0, 0]
    // This picks out the first sample of each frame window, making output
    // values directly readable as padded[frame * hop_length].
    let mut basis = vec![0.0f32; 6 * 4]; // [n_fft+2, n_fft] = [6, 4]
    basis[0] = 1.0; // filter 0 = real freq 0 = [1, 0, 0, 0]

    // padded_len = 5 + 3 = 8; n_frames = (8 - 4) / 2 + 1 = 3
    let result = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    assert_eq!(result.len(), 3 * 3); // n_freqs=3, n_frames=3

    // Freq 0 outputs (real only, imag=0, so magnitude = |real|):
    //   frame 0: padded[0] = 10.0
    //   frame 1: padded[2] = 30.0
    //   frame 2: padded[4] = 50.0  (original audio boundary)
    assert!((result[0] - 10.0).abs() < 1e-6, "frame 0: {}", result[0]);
    assert!((result[1] - 30.0).abs() < 1e-6, "frame 1: {}", result[1]);
    assert!((result[2] - 50.0).abs() < 1e-6, "frame 2: {}", result[2]);

    // Now verify reflected region: use basis [0,0,0,1] for filter 0
    // to pick up padded[frame*hop + 3].
    let mut basis2 = vec![0.0f32; 6 * 4];
    basis2[3] = 1.0; // filter 0 = [0, 0, 0, 1]

    let result2 = compute_stft_magnitude(&audio, &basis2, &params).unwrap();
    // frame 2: padded[4+3] = padded[7] = 20.0 (reflected: audio[1])
    assert!(
        (result2[2] - 20.0).abs() < 1e-6,
        "frame 2 reflected sample: expected 20.0, got {}",
        result2[2],
    );
}

#[test]
fn test_stft_n_freqs_mismatch() {
    // F1: n_freqs inconsistent with n_fft must be rejected.
    let params = StftParams {
        n_fft: 256,
        hop_length: 128,
        n_freqs: 200, // Wrong: should be 129 (256/2+1)
        pad_right: 64,
    };
    let audio = vec![0.0f32; 576];
    let basis = vec![0.0f32; 258 * 256];
    let result = compute_stft_magnitude(&audio, &basis, &params);
    assert!(
        matches!(
            result,
            Err(StftError::FreqsMismatch {
                expected: 129,
                actual: 200
            })
        ),
        "expected FreqsMismatch, got {result:?}",
    );
}

#[test]
fn test_stft_short_audio_reflection_padding_error() {
    // Audio shorter than 2 + pad_right must return an error, not panic
    // from usize underflow in the reflection index computation.
    let params = StftParams {
        n_fft: 4,
        hop_length: 2,
        n_freqs: 3,
        pad_right: 3,
    };
    // Minimum for pad_right=3 is 5 samples (2 + 3). Try 4 (one too few).
    let audio = vec![1.0f32; 4];
    let basis = vec![0.0f32; 6 * 4];
    let result = compute_stft_magnitude(&audio, &basis, &params);
    assert!(
        matches!(result, Err(StftError::AudioTooShortForPadding { .. })),
        "expected AudioTooShortForPadding, got {result:?}"
    );
}

#[test]
fn test_stft_empty_audio_reflection_padding_error() {
    // Empty audio must also error, not panic.
    let params = StftParams::default(); // pad_right=64
    let audio: Vec<f32> = vec![];
    let basis = vec![0.0f32; 258 * 256];
    let result = compute_stft_magnitude(&audio, &basis, &params);
    assert!(
        matches!(result, Err(StftError::AudioTooShortForPadding { .. })),
        "expected AudioTooShortForPadding, got {result:?}"
    );
}
