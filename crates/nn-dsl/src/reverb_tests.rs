// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for reverb filter kernels (comb + allpass).
//!
//! Part of #956 D4 (Audio DSP kernel support).

use super::*;

// --- Comb filter tests ---

fn test_comb_coeffs() -> CombCoeffs {
    CombCoeffs {
        feedback: 0.84,
        damp: 0.2,
    }
}

#[test]
fn test_comb_zero_input_zero_delay() {
    let state = CombState::new();
    let coeffs = test_comb_coeffs();
    let out = comb_process_sample_scalar(0.0, 0.0, &state, &coeffs).unwrap();
    assert_eq!(out.write_back, 0.0);
    assert_eq!(out.filterstore, 0.0);
}

#[test]
fn test_comb_feedback_path() {
    // With input=0 and a delay_read, the comb filter applies lowpass + feedback
    let state = CombState { filterstore: 0.0 };
    let coeffs = CombCoeffs {
        feedback: 0.5,
        damp: 0.0, // no damping → filterstore = delay_read
    };
    let out = comb_process_sample_scalar(0.0, 1.0, &state, &coeffs).unwrap();
    // damp=0 → damp1=1, damp2=0 → filterstore = 1.0*delay_read + 0*fs = 1.0
    assert!(
        (out.filterstore - 1.0).abs() < 1e-6,
        "filterstore = delay_read when damp=0"
    );
    // write_back = filterstore * feedback + input = 1.0 * 0.5 + 0 = 0.5
    assert!(
        (out.write_back - 0.5).abs() < 1e-6,
        "write_back = feedback * delay_read"
    );
}

#[test]
fn test_comb_full_damping() {
    // damp=1.0 → filterstore = old filterstore (ignores delay_read entirely)
    let state = CombState { filterstore: 0.5 };
    let coeffs = CombCoeffs {
        feedback: 0.8,
        damp: 1.0,
    };
    let out = comb_process_sample_scalar(0.0, 10.0, &state, &coeffs).unwrap();
    // damp1=0, damp2=1 → filterstore = 0*10 + 1*0.5 = 0.5
    assert!(
        (out.filterstore - 0.5).abs() < 1e-6,
        "full damp preserves filterstore"
    );
    // write_back = 0.5 * 0.8 + 0 = 0.4
    assert!((out.write_back - 0.4).abs() < 1e-6);
}

#[test]
fn test_comb_energy_decay() {
    // Run the comb filter in a loop with no input: energy should decay
    let coeffs = CombCoeffs {
        feedback: 0.7,
        damp: 0.3,
    };
    let mut fs = 0.0;
    let mut delay_val = 1.0; // initial impulse in delay
    for _ in 0..20 {
        let state = CombState { filterstore: fs };
        let out = comb_process_sample_scalar(0.0, delay_val, &state, &coeffs).unwrap();
        fs = out.filterstore;
        delay_val = out.write_back; // feed output back as next delay_read
    }
    assert!(
        delay_val.abs() < 0.01,
        "energy should decay with |feedback|<1, got {delay_val}"
    );
}

#[test]
fn test_comb_never_grows_without_input() {
    // With zero input and |feedback| < 1, magnitude should never increase
    let coeffs = CombCoeffs {
        feedback: 0.95,
        damp: 0.1,
    };
    let mut fs = 0.0;
    let mut delay_val: f32 = 1.0;
    let mut prev_abs = delay_val.abs();
    for _ in 0..50 {
        let state = CombState { filterstore: fs };
        let out = comb_process_sample_scalar(0.0, delay_val, &state, &coeffs).unwrap();
        fs = out.filterstore;
        delay_val = out.write_back;
        let curr_abs = delay_val.abs();
        // Allow small floating-point overshoot
        assert!(
            curr_abs <= prev_abs + 1e-6,
            "energy grew: prev={prev_abs}, curr={curr_abs}"
        );
        prev_abs = curr_abs;
    }
}

// --- Comb config validation ---

#[test]
fn test_validate_comb_valid() {
    assert!(validate_comb_config(&test_comb_coeffs()).is_ok());
}

#[test]
fn test_validate_comb_feedback_at_boundary() {
    let coeffs = CombCoeffs {
        feedback: 1.0,
        damp: 0.2,
    };
    assert!(validate_comb_config(&coeffs).is_err());
}

#[test]
fn test_validate_comb_negative_feedback() {
    // Negative feedback is valid as long as |feedback| < 1
    let coeffs = CombCoeffs {
        feedback: -0.5,
        damp: 0.2,
    };
    assert!(validate_comb_config(&coeffs).is_ok());
}

#[test]
fn test_validate_comb_bad_damp() {
    let coeffs = CombCoeffs {
        feedback: 0.5,
        damp: -0.1,
    };
    assert!(validate_comb_config(&coeffs).is_err());

    let coeffs2 = CombCoeffs {
        feedback: 0.5,
        damp: 1.1,
    };
    assert!(validate_comb_config(&coeffs2).is_err());
}

#[test]
fn test_comb_reject_nan() {
    let state = CombState::new();
    let coeffs = test_comb_coeffs();
    assert!(comb_process_sample_scalar(f32::NAN, 0.0, &state, &coeffs).is_err());
    assert!(comb_process_sample_scalar(0.0, f32::NAN, &state, &coeffs).is_err());
}

#[test]
fn test_comb_reject_inf() {
    let state = CombState::new();
    let coeffs = test_comb_coeffs();
    assert!(comb_process_sample_scalar(f32::INFINITY, 0.0, &state, &coeffs).is_err());
}

// --- Allpass filter tests ---

fn test_allpass_coeffs() -> AllpassCoeffs {
    AllpassCoeffs { feedback: 0.5 }
}

#[test]
fn test_allpass_zero_input_zero_delay() {
    let coeffs = test_allpass_coeffs();
    let out = allpass_process_sample_scalar(0.0, 0.0, &coeffs).unwrap();
    assert_eq!(out.y, 0.0);
    assert_eq!(out.write_back, 0.0);
}

#[test]
fn test_allpass_impulse_response() {
    // First sample: input=1, delay_read=0 (empty delay line)
    let coeffs = AllpassCoeffs { feedback: 0.5 };
    let out = allpass_process_sample_scalar(1.0, 0.0, &coeffs).unwrap();
    // write_back = 1.0 + 0.5*0.0 = 1.0
    assert!((out.write_back - 1.0).abs() < 1e-6);
    // y = 0.0 - 0.5*1.0 = -0.5
    assert!((out.y - (-0.5)).abs() < 1e-6, "got y={}", out.y);
}

#[test]
fn test_allpass_energy_preservation() {
    // The allpass filter preserves energy (|H(z)| = 1 at all frequencies).
    // For a finite sequence, total energy in ≈ total energy out.
    let coeffs = AllpassCoeffs { feedback: 0.5 };
    let input_signal = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut delay_buf = [0.0_f32; 8];
    let mut write_pos = 0;
    let mut read_pos = 1; // 1-sample delay
    let mut output_energy = 0.0_f32;

    for &inp in &input_signal {
        let delay_read = delay_buf[read_pos % delay_buf.len()];
        let out = allpass_process_sample_scalar(inp, delay_read, &coeffs).unwrap();
        let len = delay_buf.len();
        delay_buf[write_pos % len] = out.write_back;
        output_energy += out.y * out.y;
        write_pos += 1;
        read_pos += 1;
    }

    // Drain remaining energy from delay line
    for _ in 0..16 {
        let delay_read = delay_buf[read_pos % delay_buf.len()];
        let out = allpass_process_sample_scalar(0.0, delay_read, &coeffs).unwrap();
        let len = delay_buf.len();
        delay_buf[write_pos % len] = out.write_back;
        output_energy += out.y * out.y;
        write_pos += 1;
        read_pos += 1;
    }

    let input_energy: f32 = input_signal.iter().map(|x| x * x).sum();
    // Allow some tolerance for finite-length analysis
    assert!(
        (output_energy - input_energy).abs() < 0.1,
        "energy mismatch: in={input_energy}, out={output_energy}"
    );
}

#[test]
fn test_allpass_output_bounded() {
    // For bounded inputs, allpass output should not exceed input bounds
    let coeffs = AllpassCoeffs { feedback: 0.7 };
    let bound = 5.0;
    for &inp in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
        for &dr in &[-5.0, -1.0, 0.0, 1.0, 5.0] {
            let out = allpass_process_sample_scalar(inp, dr, &coeffs).unwrap();
            // Output should be bounded, though the exact bound depends on history
            assert!(
                out.y.is_finite(),
                "inp={inp}, dr={dr}: output not finite: {}",
                out.y
            );
            // Loose bound: |y| ≤ (1 + |feedback|) * max(|input|, |delay_read|)
            let loose_bound = (1.0 + coeffs.feedback.abs()) * bound;
            assert!(
                out.y.abs() <= loose_bound + 1e-6,
                "inp={inp}, dr={dr}: |y|={} > loose_bound={loose_bound}",
                out.y.abs()
            );
        }
    }
}

// --- Allpass config validation ---

#[test]
fn test_validate_allpass_valid() {
    assert!(validate_allpass_config(&test_allpass_coeffs()).is_ok());
}

#[test]
fn test_validate_allpass_at_boundary() {
    let coeffs = AllpassCoeffs { feedback: 1.0 };
    assert!(validate_allpass_config(&coeffs).is_err());
}

#[test]
fn test_validate_allpass_negative_feedback() {
    let coeffs = AllpassCoeffs { feedback: -0.7 };
    assert!(validate_allpass_config(&coeffs).is_ok());
}

#[test]
fn test_allpass_reject_nan() {
    let coeffs = test_allpass_coeffs();
    assert!(allpass_process_sample_scalar(f32::NAN, 0.0, &coeffs).is_err());
    assert!(allpass_process_sample_scalar(0.0, f32::NAN, &coeffs).is_err());
}

#[test]
fn test_allpass_reject_inf() {
    let coeffs = test_allpass_coeffs();
    assert!(allpass_process_sample_scalar(f32::INFINITY, 0.0, &coeffs).is_err());
}

// --- Default state ---

#[test]
fn test_comb_default_state() {
    let state = CombState::default();
    assert_eq!(state.filterstore, 0.0);
}
