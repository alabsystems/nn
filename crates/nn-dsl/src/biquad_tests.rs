// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for biquad IIR filter kernel.
//!
//! Part of #956 (Audio DSP kernel support).

use super::*;

// --- Peaking EQ tests ---

#[test]
fn test_peaking_0db_is_transparent() {
    // 0 dB gain → A=1 → identical numerator/denominator → H(z) = 1
    // Coefficients: b0=1, b1=a1, b2=a2 (transparent, not necessarily identity)
    let coeffs = biquad_peaking(44100.0, 1000.0, 0.0, 0.707).unwrap();
    assert!(
        coeffs.is_transparent(1e-5),
        "0 dB peaking must be transparent (H(z)=1), got {coeffs:?}"
    );
    // Verify dc_gain = 1.0 (consequence of transparency)
    let dc = coeffs
        .dc_gain()
        .expect("transparent filter has finite DC gain");
    assert!((dc - 1.0).abs() < 1e-5, "DC gain must be 1.0, got {dc}");
}

#[test]
fn test_peaking_positive_gain_stable() {
    let coeffs = biquad_peaking(44100.0, 1000.0, 6.0, 1.0).unwrap();
    assert!(coeffs.is_stable(), "peaking +6dB must be stable");
    // DC gain should be ~1.0 for peaking (only affects near center freq)
    let dc = coeffs.dc_gain().unwrap();
    assert!(
        (dc - 1.0).abs() < 0.1,
        "peaking DC gain should be near 1.0, got {dc}"
    );
}

#[test]
fn test_peaking_negative_gain_stable() {
    let coeffs = biquad_peaking(44100.0, 1000.0, -12.0, 0.5).unwrap();
    assert!(coeffs.is_stable(), "peaking -12dB must be stable");
}

#[test]
fn test_peaking_at_various_frequencies() {
    for freq in [100.0, 500.0, 1000.0, 5000.0, 10000.0] {
        let coeffs = biquad_peaking(44100.0, freq, 3.0, 1.0).unwrap();
        assert!(coeffs.is_stable(), "peaking at {freq}Hz must be stable");
    }
}

// --- High-shelf tests ---

#[test]
fn test_high_shelf_0db_is_transparent() {
    let coeffs = biquad_high_shelf(44100.0, 1000.0, 0.0, 0.707).unwrap();
    assert!(
        coeffs.is_transparent(1e-4),
        "0 dB high-shelf must be transparent (H(z)=1), got {coeffs:?}"
    );
}

#[test]
fn test_high_shelf_positive_gain_boosts_high() {
    let coeffs = biquad_high_shelf(44100.0, 4000.0, 6.0, 0.707).unwrap();
    assert!(coeffs.is_stable(), "high-shelf +6dB must be stable");
    // Nyquist gain should be boosted
    let nyq = coeffs.nyquist_gain().unwrap();
    assert!(
        nyq > 1.5,
        "high-shelf +6dB Nyquist gain should be > 1.5, got {nyq}"
    );
}

#[test]
fn test_high_shelf_negative_gain_cuts_high() {
    let coeffs = biquad_high_shelf(44100.0, 4000.0, -6.0, 0.707).unwrap();
    assert!(coeffs.is_stable());
    let nyq = coeffs.nyquist_gain().unwrap();
    assert!(
        nyq < 0.7,
        "high-shelf -6dB Nyquist gain should be < 0.7, got {nyq}"
    );
}

// --- Bandpass tests ---

#[test]
fn test_bandpass_coefficients() {
    let coeffs = biquad_bandpass(44100.0, 1000.0, 1.0).unwrap();
    assert!(coeffs.is_stable(), "bandpass must be stable");
    // Bandpass has zero DC gain: b0 + b1 + b2 = alpha + 0 + (-alpha) = 0
    let dc = coeffs.dc_gain().unwrap();
    assert!(dc.abs() < 1e-5, "bandpass DC gain must be ~0, got {dc}");
}

#[test]
fn test_bandpass_zero_nyquist() {
    let coeffs = biquad_bandpass(44100.0, 1000.0, 1.0).unwrap();
    // Nyquist gain: (b0 - b1 + b2) / (1 - a1 + a2) = (alpha - 0 - alpha) / ... = 0
    let nyq = coeffs.nyquist_gain().unwrap();
    assert!(
        nyq.abs() < 1e-5,
        "bandpass Nyquist gain must be ~0, got {nyq}"
    );
}

// --- Process sample tests ---

#[test]
fn test_process_sample_identity() {
    let coeffs = BiquadCoeffs {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    };
    let out = biquad_process_sample_scalar(0.5, &coeffs, 0.0, 0.0).unwrap();
    assert_eq!(out.y, 0.5);
    assert_eq!(out.z1, 0.0);
    assert_eq!(out.z2, 0.0);
}

#[test]
fn test_process_sample_sequence() {
    // FIR filter: b=[0.5, 0.3, 0.2], a=[0, 0]
    let coeffs = BiquadCoeffs {
        b0: 0.5,
        b1: 0.3,
        b2: 0.2,
        a1: 0.0,
        a2: 0.0,
    };

    // Step 1: x=1.0, z1=0, z2=0
    let out1 = biquad_process_sample_scalar(1.0, &coeffs, 0.0, 0.0).unwrap();
    assert!((out1.y - 0.5).abs() < 1e-6, "step 1 output");

    // Step 2: x=0.0, z1=out1.z1, z2=out1.z2
    let out2 = biquad_process_sample_scalar(0.0, &coeffs, out1.z1, out1.z2).unwrap();
    assert!((out2.y - 0.3).abs() < 1e-6, "step 2 output = b1*x_prev");

    // Step 3: x=0.0
    let out3 = biquad_process_sample_scalar(0.0, &coeffs, out2.z1, out2.z2).unwrap();
    assert!(
        (out3.y - 0.2).abs() < 1e-6,
        "step 3 output = b2*x_prev_prev"
    );
}

#[test]
fn test_process_sample_with_feedback() {
    // Simple IIR: b=[1, 0, 0], a=[-0.5, 0] (first-order pole at z=0.5)
    let coeffs = BiquadCoeffs {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: -0.5,
        a2: 0.0,
    };
    assert!(coeffs.is_stable());

    // Impulse response: y[0]=1, y[1]=0.5, y[2]=0.25, ...
    let out0 = biquad_process_sample_scalar(1.0, &coeffs, 0.0, 0.0).unwrap();
    assert!((out0.y - 1.0).abs() < 1e-6);

    let out1 = biquad_process_sample_scalar(0.0, &coeffs, out0.z1, out0.z2).unwrap();
    assert!((out1.y - 0.5).abs() < 1e-6);

    let out2 = biquad_process_sample_scalar(0.0, &coeffs, out1.z1, out1.z2).unwrap();
    assert!((out2.y - 0.25).abs() < 1e-6);
}

// --- Stability tests ---

#[test]
fn test_stability_valid() {
    // Stable: a1=0, a2=0.5 → |0.5|<1, 1+0+0.5>0, 1-0+0.5>0
    let c = BiquadCoeffs {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.5,
    };
    assert!(c.is_stable());
}

#[test]
fn test_stability_invalid_a2_too_large() {
    let c = BiquadCoeffs {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 1.0,
    };
    assert!(!c.is_stable());
}

#[test]
fn test_stability_invalid_jury_condition_2() {
    // 1 + a1 + a2 = 1 + (-3) + 1 = -1 < 0
    let c = BiquadCoeffs {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: -3.0,
        a2: 1.0,
    };
    assert!(!c.is_stable());
}

// --- Error handling tests ---

#[test]
fn test_reject_nan_input() {
    let coeffs = BiquadCoeffs {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    };
    let result = biquad_process_sample_scalar(f32::NAN, &coeffs, 0.0, 0.0);
    assert!(result.is_err());
}

#[test]
fn test_reject_inf_coefficient() {
    let coeffs = BiquadCoeffs {
        b0: f32::INFINITY,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    };
    let result = biquad_process_sample_scalar(1.0, &coeffs, 0.0, 0.0);
    assert!(result.is_err());
}

#[test]
fn test_reject_negative_sample_rate() {
    let result = biquad_peaking(-44100.0, 1000.0, 0.0, 1.0);
    assert!(result.is_err());
}

#[test]
fn test_reject_freq_above_nyquist() {
    let result = biquad_peaking(44100.0, 25000.0, 0.0, 1.0);
    assert!(result.is_err());
}

#[test]
fn test_reject_zero_q() {
    let result = biquad_peaking(44100.0, 1000.0, 0.0, 0.0);
    assert!(result.is_err());
}

#[test]
fn test_reject_negative_freq() {
    let result = biquad_bandpass(44100.0, -100.0, 1.0);
    assert!(result.is_err());
}

// --- DC/Nyquist response tests ---

#[test]
fn test_peaking_dc_gain_unity() {
    // Peaking EQ at 1kHz with +6dB should have DC gain ~1.0
    let coeffs = biquad_peaking(44100.0, 1000.0, 6.0, 1.0).unwrap();
    let dc = coeffs.dc_gain().unwrap();
    assert!(
        (dc - 1.0).abs() < 0.05,
        "peaking DC gain near unity, got {dc}"
    );
}

#[test]
fn test_high_shelf_dc_gain_near_unity() {
    // High-shelf at 4kHz boosts above 4kHz; DC should be ~1.0
    let coeffs = biquad_high_shelf(44100.0, 4000.0, 6.0, 0.707).unwrap();
    let dc = coeffs.dc_gain().unwrap();
    assert!(
        (dc - 1.0).abs() < 0.1,
        "high-shelf DC gain near unity, got {dc}"
    );
}

// --- Identity detection ---

#[test]
fn test_is_identity_true() {
    let c = BiquadCoeffs {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    };
    assert!(c.is_identity(1e-6));
}

#[test]
fn test_is_identity_false() {
    let c = BiquadCoeffs {
        b0: 0.5,
        b1: 0.3,
        b2: 0.2,
        a1: 0.0,
        a2: 0.0,
    };
    assert!(!c.is_identity(1e-6));
}
