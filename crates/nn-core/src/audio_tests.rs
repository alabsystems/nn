// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for [`crate::audio`] — mel-frequency conversions, Hann
//! windowing, and crossfade blending.

use crate::audio::*;

// -- HTK mel scale edge cases -------------------------------------------------

#[test]
fn test_htk_mel_monotonic_increasing() {
    // Hz-to-mel must be monotonically increasing.
    let freqs = [0.0, 100.0, 440.0, 1000.0, 4000.0, 8000.0, 16000.0, 44100.0];
    let mels: Vec<f64> = freqs.iter().map(|&hz| hz_to_mel_htk(hz)).collect();
    for i in 1..mels.len() {
        assert!(
            mels[i] > mels[i - 1],
            "mel({}) = {} should be > mel({}) = {}",
            freqs[i],
            mels[i],
            freqs[i - 1],
            mels[i - 1]
        );
    }
}

#[test]
fn test_htk_mel_zero_is_zero() {
    assert_eq!(hz_to_mel_htk(0.0), 0.0);
    assert_eq!(mel_to_hz_htk(0.0), 0.0);
}

#[test]
fn test_htk_mel_large_frequency() {
    // Test at ultrasonic frequencies to ensure no overflow
    let hz = 96000.0;
    let mel = hz_to_mel_htk(hz);
    let back = mel_to_hz_htk(mel);
    assert!((back - hz).abs() < 1e-6, "roundtrip failed at {hz} Hz");
    assert!(mel.is_finite());
}

// -- Slaney mel scale edge cases ----------------------------------------------

#[test]
fn test_slaney_mel_monotonic_increasing() {
    let freqs = [0.0, 100.0, 500.0, 999.0, 1000.0, 1001.0, 4000.0, 16000.0];
    let mels: Vec<f64> = freqs.iter().map(|&hz| hz_to_mel_slaney(hz)).collect();
    for i in 1..mels.len() {
        assert!(
            mels[i] > mels[i - 1],
            "slaney mel({}) = {} should be > slaney mel({}) = {}",
            freqs[i],
            mels[i],
            freqs[i - 1],
            mels[i - 1]
        );
    }
}

#[test]
fn test_slaney_mel_log_region() {
    // Above 1 kHz, the Slaney scale is logarithmic. Check a known value.
    let hz = 2000.0;
    let mel = hz_to_mel_slaney(hz);
    // Should be > 15.0 (the mel at 1000 Hz)
    assert!(
        mel > 15.0,
        "slaney mel at 2000 Hz should be > 15, got {mel}"
    );
    // Roundtrip
    let back = mel_to_hz_slaney(mel);
    assert!(
        (back - hz).abs() < 1e-9,
        "slaney roundtrip failed for {hz} Hz: got {back}"
    );
}

#[test]
fn test_slaney_mel_zero_hz() {
    assert_eq!(hz_to_mel_slaney(0.0), 0.0);
    assert_eq!(mel_to_hz_slaney(0.0), 0.0);
}

#[test]
fn test_slaney_mel_continuity_at_boundary() {
    // The piecewise function should be continuous at 1000 Hz.
    // Approach from slightly below and slightly above.
    let below = hz_to_mel_slaney(999.999);
    let at = hz_to_mel_slaney(1000.0);
    let above = hz_to_mel_slaney(1000.001);
    assert!(
        (below - at).abs() < 0.01,
        "discontinuity at 1 kHz boundary: below={below}, at={at}"
    );
    assert!(
        (above - at).abs() < 0.01,
        "discontinuity at 1 kHz boundary: at={at}, above={above}"
    );
}

#[test]
fn test_htk_vs_slaney_at_1khz() {
    // Both scales have mel=15 at Hz=1000 for Slaney, and mel~1000 for HTK.
    // They should diverge significantly at 1 kHz.
    let htk = hz_to_mel_htk(1000.0);
    let slaney = hz_to_mel_slaney(1000.0);
    // HTK mel at 1 kHz is approximately 999.985 (2595 * log10(1 + 1000/700))
    assert!((htk - 999.985).abs() < 0.1);
    // Slaney mel at 1 kHz is exactly 15.0
    assert!((slaney - 15.0).abs() < 1e-12);
}

// -- Hann window extended tests -----------------------------------------------

#[test]
fn test_hann_window_length_one() {
    let w = hann_window(1);
    assert_eq!(w.len(), 1);
    // w[0] = 0.5 * (1 - cos(0)) = 0
    assert!(w[0].abs() < 1e-15);
}

#[test]
fn test_hann_window_length_two() {
    let w = hann_window(2);
    assert_eq!(w.len(), 2);
    // w[0] = 0.5 * (1 - cos(0)) = 0
    // w[1] = 0.5 * (1 - cos(pi)) = 1
    assert!(w[0].abs() < 1e-15);
    assert!((w[1] - 1.0).abs() < 1e-15);
}

#[test]
fn test_hann_window_peak_position() {
    // For even n, peak is at n/2. For n=256, w[128]=1.0
    let w = hann_window(256);
    assert!((w[128] - 1.0).abs() < 1e-15);
}

#[test]
fn test_hann_window_sum_constant_overlap_add() {
    // Hann windows with 50% overlap sum to a constant (COLA property).
    // For a periodic Hann window of length n, w[i] + w[i + n/2] = 1.
    let n = 128;
    let w = hann_window(n);
    let half = n / 2;
    for i in 0..half {
        let sum = w[i] + w[i + half];
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "COLA violation at i={i}: w[{i}]={} + w[{}]={} = {sum}",
            w[i],
            i + half,
            w[i + half]
        );
    }
}

// -- Crossfade extended tests -------------------------------------------------

#[test]
fn test_crossfade_linear_blend_negative_values() {
    let tail = vec![-1.0_f32; 4];
    let head = vec![1.0_f32; 4];
    let result = crossfade_linear_blend(&tail, &head, 4);
    // alpha goes 0, 1/3, 2/3, 1
    // result: -1*(1-0)+1*0=-1, -1*(2/3)+1*(1/3)=-1/3, -1*(1/3)+1*(2/3)=1/3, -1*0+1*1=1
    assert!((result[0] - (-1.0)).abs() < 1e-6);
    assert!((result[3] - 1.0).abs() < 1e-6);
    // Midpoints should be symmetric around 0
    assert!((result[1] + result[2]).abs() < 1e-6);
}

#[test]
fn test_crossfade_linear_blend_large_window() {
    // Verify blend works for a 1000-sample window without panicking.
    let n = 1000;
    let tail = vec![1.0_f32; n];
    let head = vec![0.0_f32; n];
    let result = crossfade_linear_blend(&tail, &head, n);
    assert_eq!(result.len(), n);
    // First and last samples
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!(result[n - 1].abs() < 1e-6);
    // Monotonically decreasing (tail=1, head=0 => result decreases)
    for i in 1..n {
        assert!(
            result[i] <= result[i - 1] + 1e-6,
            "not monotonic at i={i}: {} > {}",
            result[i],
            result[i - 1]
        );
    }
}

#[test]
fn test_crossfade_blend_into_cf_one() {
    // cf=1 is a special case: returns average of tail[0] and head[0].
    let mut out = Vec::new();
    let tail = [2.0_f32];
    let head = [4.0_f32];
    crossfade_blend_into(&mut out, &tail, &head, 1, 1);
    assert_eq!(out.len(), 1);
    assert!((out[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_crossfade_blend_into_consistent_with_linear_blend() {
    // The two crossfade functions should produce identical results
    // when cf == count == limit.
    let n = 8;
    let tail: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let head: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 * 0.1).collect();

    let direct = crossfade_linear_blend(&tail, &head, n);

    let mut into = Vec::new();
    crossfade_blend_into(&mut into, &tail, &head, n, n);

    assert_eq!(direct.len(), into.len());
    for (i, (a, b)) in direct.iter().zip(into.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "mismatch at i={i}: direct={a}, into={b}"
        );
    }
}

#[test]
fn test_crossfade_blend_into_limit_greater_than_cf() {
    // When limit > cf, n = cf.min(limit) = cf.
    let mut out = Vec::new();
    let tail = vec![1.0_f32; 3];
    let head = vec![0.0_f32; 3];
    crossfade_blend_into(&mut out, &tail, &head, 3, 100);
    assert_eq!(out.len(), 3);
}
