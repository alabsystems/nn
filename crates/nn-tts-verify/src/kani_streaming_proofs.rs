// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

/// Prove: validate rejects zero crossfade_samples.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn streaming_config_rejects_zero_crossfade() {
    let config = crate::streaming::StreamingConfig {
        crossfade_samples: 0,
        ..crate::streaming::StreamingConfig::default()
    };
    assert!(
        config.validate().is_err(),
        "zero crossfade_samples must be rejected"
    );
}

/// Prove: validate rejects margin_samples < crossfade_samples.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn streaming_config_rejects_margin_less_than_crossfade() {
    let crossfade: usize = kani::any();
    let margin: usize = kani::any();
    kani::assume(crossfade > 0 && crossfade <= 10000);
    kani::assume(margin < crossfade);

    let config = crate::streaming::StreamingConfig {
        sample_rate: 24000,
        crossfade_samples: crossfade,
        margin_samples: margin,
        click_threshold: 0.3,
        energy_lo: 0.5,
        energy_hi: 1.5,
        spectral_threshold: 0.15,
    };
    assert!(
        config.validate().is_err(),
        "margin < crossfade must be rejected"
    );
}

/// Prove: validate accepts margin_samples == crossfade_samples.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn streaming_config_accepts_margin_equals_crossfade() {
    let n: usize = kani::any();
    kani::assume(n > 0 && n <= 10000);

    let config = crate::streaming::StreamingConfig {
        sample_rate: 24000,
        crossfade_samples: n,
        margin_samples: n,
        click_threshold: 0.3,
        energy_lo: 0.5,
        energy_hi: 1.5,
        spectral_threshold: 0.15,
    };
    assert!(
        config.validate().is_ok(),
        "margin == crossfade must be accepted"
    );
}

/// Prove: validate rejects NaN click_threshold.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn streaming_config_rejects_nan_click_threshold() {
    let config = crate::streaming::StreamingConfig {
        click_threshold: f64::NAN,
        ..crate::streaming::StreamingConfig::default()
    };
    assert!(
        config.validate().is_err(),
        "NaN click_threshold must be rejected"
    );
}

/// Prove: validate rejects Inf spectral_threshold.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn streaming_config_rejects_inf_spectral_threshold() {
    let config = crate::streaming::StreamingConfig {
        spectral_threshold: f64::INFINITY,
        ..crate::streaming::StreamingConfig::default()
    };
    assert!(
        config.validate().is_err(),
        "Inf spectral_threshold must be rejected"
    );
}

/// Prove: validate rejects negative energy_lo.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn streaming_config_rejects_negative_energy_lo() {
    let val: f64 = kani::any();
    kani::assume(val.is_finite() && val <= 0.0);

    let config = crate::streaming::StreamingConfig {
        energy_lo: val,
        ..crate::streaming::StreamingConfig::default()
    };
    assert!(
        config.validate().is_err(),
        "non-positive energy_lo must be rejected"
    );
}

/// Prove: validate rejects energy_lo >= energy_hi (inverted range).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn streaming_config_rejects_inverted_energy_range() {
    let lo: f64 = kani::any();
    let hi: f64 = kani::any();
    kani::assume(lo.is_finite() && hi.is_finite());
    kani::assume(lo > 0.0 && hi > 0.0);
    kani::assume(lo >= hi);

    let config = crate::streaming::StreamingConfig {
        energy_lo: lo,
        energy_hi: hi,
        ..crate::streaming::StreamingConfig::default()
    };
    assert!(
        config.validate().is_err(),
        "energy_lo >= energy_hi must be rejected"
    );
}

// ---- crossfade_linear proofs ------------------------------------------------

/// Prove: crossfade_linear endpoints — first sample is tail[0], last is head[last].
///
/// At alpha=0 (i=0): out = tail[0]*(1-0) + head[0]*0 = tail[0].
/// At alpha=1 (i=n-1): out = tail[n-1]*(1-1) + head[n-1]*1 = head[n-1].
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(4)]
fn crossfade_endpoints_correct() {
    let t0: f32 = kani::any();
    let t1: f32 = kani::any();
    let h0: f32 = kani::any();
    let h1: f32 = kani::any();
    kani::assume(t0.is_finite() && t1.is_finite());
    kani::assume(h0.is_finite() && h1.is_finite());
    kani::assume(t0.abs() <= 1.0 && t1.abs() <= 1.0);
    kani::assume(h0.abs() <= 1.0 && h1.abs() <= 1.0);

    let tail = [t0, t1];
    let head = [h0, h1];
    let result = crate::streaming::crossfade_linear(&tail, &head).unwrap();

    // First sample: alpha=0 → tail[0]
    assert!(
        (f64::from(result[0]) - f64::from(t0)).abs() < 1e-6,
        "first sample must equal tail[0]"
    );
    // Last sample: alpha=1 → head[1]
    assert!(
        (f64::from(result[1]) - f64::from(h1)).abs() < 1e-6,
        "last sample must equal head[last]"
    );
}

/// Prove: crossfade_linear of identical signals returns that signal.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn crossfade_identical_signals_identity() {
    let a: f32 = kani::any();
    let b: f32 = kani::any();
    kani::assume(a.is_finite() && b.is_finite());
    kani::assume(a.abs() <= 1.0 && b.abs() <= 1.0);

    let signal = [a, b];
    let result = crate::streaming::crossfade_linear(&signal, &signal).unwrap();

    for i in 0..2 {
        assert!(
            (f64::from(result[i]) - f64::from(signal[i])).abs() < 1e-6,
            "crossfade of identical signals must return that signal at index {i}"
        );
    }
}

/// Prove: crossfade_linear output is finite when inputs are finite and bounded.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(4)]
fn crossfade_output_is_finite() {
    let t0: f32 = kani::any();
    let t1: f32 = kani::any();
    let h0: f32 = kani::any();
    let h1: f32 = kani::any();
    kani::assume(t0.is_finite() && t1.is_finite());
    kani::assume(h0.is_finite() && h1.is_finite());
    kani::assume(t0.abs() <= 1.0 && t1.abs() <= 1.0);
    kani::assume(h0.abs() <= 1.0 && h1.abs() <= 1.0);

    let result = crate::streaming::crossfade_linear(&[t0, t1], &[h0, h1]).unwrap();
    for i in 0..2 {
        assert!(
            result[i].is_finite(),
            "crossfade output must be finite at index {i}"
        );
    }
}

/// Prove: crossfade_linear empty inputs produce empty output (no panic).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn crossfade_empty_inputs_empty_output() {
    let empty: &[f32] = &[];
    let result = crate::streaming::crossfade_linear(empty, empty).unwrap();
    assert!(result.is_empty(), "empty inputs must produce empty output");
}

/// Prove: crossfade_linear single sample returns head (n <= 1 path).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(2)]
fn crossfade_single_sample_returns_head() {
    let t: f32 = kani::any();
    let h: f32 = kani::any();
    kani::assume(t.is_finite() && h.is_finite());

    let result = crate::streaming::crossfade_linear(&[t], &[h]).unwrap();
    assert_eq!(result.len(), 1);
    assert!(
        (f64::from(result[0]) - f64::from(h)).abs() < 1e-6,
        "single sample crossfade must return head"
    );
}
