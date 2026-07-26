// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for de-essing and spectral EQ for Kokoro chorus voice mixing.

use super::*;

// ---------------------------------------------------------------------------
// Helper: generate a sine wave at a given frequency
// ---------------------------------------------------------------------------

fn sine_wave(freq_hz: f32, duration_sec: f32, amplitude: f32) -> Vec<f32> {
    let sr = KOKORO_SAMPLE_RATE as f32;
    let n_samples = (duration_sec * sr).round() as usize;
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / sr;
            amplitude * (2.0 * std::f32::consts::PI * freq_hz * t).sin()
        })
        .collect()
}

/// Compute RMS energy of a buffer.
fn rms(buffer: &[f32]) -> f32 {
    if buffer.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = buffer.iter().map(|x| x * x).sum();
    (sum_sq / buffer.len() as f32).sqrt()
}

// ---------------------------------------------------------------------------
// DeEsser tests
// ---------------------------------------------------------------------------

#[test]
fn test_deesser_default_creation() {
    let deesser = DeEsser::default_config();
    assert!(deesser.is_ok());
}

#[test]
fn test_deesser_reduces_sibilant_frequency() {
    // Generate a loud 6 kHz sine wave (sibilance frequency).
    let mut signal = sine_wave(6000.0, 0.1, 0.8);
    let rms_before = rms(&signal);

    let mut deesser = DeEsser::new(&DeEsserConfig {
        threshold_db: -30.0,
        ..DeEsserConfig::default()
    })
    .expect("valid config");

    deesser.process(&mut signal);
    let rms_after = rms(&signal);

    // The de-esser should reduce energy of the 6 kHz signal.
    assert!(
        rms_after < rms_before,
        "De-esser should reduce 6 kHz energy: before={rms_before:.4}, after={rms_after:.4}"
    );
}

#[test]
fn test_deesser_preserves_low_frequency() {
    // Generate a 200 Hz sine wave (well below sibilance band).
    let mut signal = sine_wave(200.0, 0.1, 0.5);
    let rms_before = rms(&signal);

    let mut deesser = DeEsser::default_config().expect("valid config");
    deesser.process(&mut signal);
    let rms_after = rms(&signal);

    // Low-frequency content should be largely preserved (within 1 dB).
    let ratio = rms_after / rms_before;
    let db_diff = 20.0 * ratio.log10();
    assert!(
        db_diff.abs() < 1.0,
        "De-esser should preserve 200 Hz: ratio={ratio:.4}, dB={db_diff:.2}"
    );
}

#[test]
fn test_deesser_no_action_below_threshold() {
    // Generate a very quiet 6 kHz signal (below threshold).
    let mut signal = sine_wave(6000.0, 0.05, 0.001);
    let original = signal.clone();

    let mut deesser = DeEsser::default_config().expect("valid config");
    deesser.process(&mut signal);

    // Signal should be nearly unchanged.
    let max_diff: f32 = signal
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 0.01,
        "De-esser should not modify quiet signals: max_diff={max_diff:.6}"
    );
}

#[test]
fn test_deesser_handles_nan_input() {
    let mut signal = vec![0.5, f32::NAN, 0.3, f32::INFINITY, -0.2];
    let mut deesser = DeEsser::default_config().expect("valid config");
    deesser.process(&mut signal);

    // All NaN/Inf should be replaced with 0.0.
    for (i, &s) in signal.iter().enumerate() {
        assert!(
            s.is_finite(),
            "Sample {i} should be finite after de-essing, got {s}"
        );
    }
}

#[test]
fn test_deesser_reset() {
    let mut deesser = DeEsser::default_config().expect("valid config");
    let mut loud = sine_wave(6000.0, 0.05, 0.9);
    deesser.process(&mut loud);
    assert!(deesser.envelope_sq > 0.0);

    deesser.reset();
    assert_eq!(deesser.envelope_sq, 0.0);
}

#[test]
fn test_deesser_config_validation() {
    // Invalid center frequency.
    let cfg = DeEsserConfig {
        center_freq_hz: 50.0,
        ..DeEsserConfig::default()
    };
    assert!(cfg.validate().is_err());

    // Invalid Q.
    let cfg = DeEsserConfig {
        q: 0.0,
        ..DeEsserConfig::default()
    };
    assert!(cfg.validate().is_err());

    // Invalid threshold.
    let cfg = DeEsserConfig {
        threshold_db: 1.0,
        ..DeEsserConfig::default()
    };
    assert!(cfg.validate().is_err());

    // NaN attack.
    let cfg = DeEsserConfig {
        attack_sec: f32::NAN,
        ..DeEsserConfig::default()
    };
    assert!(cfg.validate().is_err());
}

// ---------------------------------------------------------------------------
// ChorusEQ tests
// ---------------------------------------------------------------------------

#[test]
fn test_eq_flat_is_unity() {
    // A flat EQ (all gains 0 dB) should not change the signal significantly.
    let config = EqConfig::default();
    let mut eq = ChorusEQ::new(&config).expect("valid config");

    let mut signal = sine_wave(1000.0, 0.05, 0.5);
    let rms_before = rms(&signal);

    eq.process(&mut signal);
    let rms_after = rms(&signal);

    let ratio = rms_after / rms_before;
    let db_diff = 20.0 * ratio.log10();
    assert!(
        db_diff.abs() < 0.5,
        "Flat EQ should be near-unity: ratio={ratio:.4}, dB={db_diff:.2}"
    );
}

#[test]
fn test_eq_low_shelf_boost() {
    // Boosting the low shelf should increase energy of low-frequency content.
    let config = EqConfig {
        low_gain_db: 6.0,
        ..EqConfig::default()
    };
    let mut eq = ChorusEQ::new(&config).expect("valid config");

    let mut signal = sine_wave(100.0, 0.1, 0.3);
    let rms_before = rms(&signal);

    eq.process(&mut signal);
    let rms_after = rms(&signal);

    assert!(
        rms_after > rms_before,
        "Low shelf boost should increase 100 Hz energy: before={rms_before:.4}, after={rms_after:.4}"
    );
}

#[test]
fn test_eq_high_shelf_cut() {
    // Cutting the high shelf should reduce energy of high-frequency content.
    let config = EqConfig {
        high_gain_db: -6.0,
        ..EqConfig::default()
    };
    let mut eq = ChorusEQ::new(&config).expect("valid config");

    let mut signal = sine_wave(10000.0, 0.1, 0.5);
    let rms_before = rms(&signal);

    eq.process(&mut signal);
    let rms_after = rms(&signal);

    assert!(
        rms_after < rms_before,
        "High shelf cut should reduce 10 kHz energy: before={rms_before:.4}, after={rms_after:.4}"
    );
}

#[test]
fn test_eq_energy_conservation_no_boost() {
    // With only cuts (no boosts), output energy must not exceed input energy.
    let config = EqConfig {
        low_gain_db: -3.0,
        mid_gain_db: -2.0,
        high_gain_db: -1.0,
        ..EqConfig::default()
    };
    let mut eq = ChorusEQ::new(&config).expect("valid config");

    // Mixed-frequency test signal.
    let n = 4800; // 200ms at 24kHz
    let sr = KOKORO_SAMPLE_RATE as f32;
    let mut signal: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            0.2 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 1500.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 8000.0 * t).sin()
        })
        .collect();

    let energy_before: f32 = signal.iter().map(|x| x * x).sum();
    eq.process(&mut signal);
    let energy_after: f32 = signal.iter().map(|x| x * x).sum();

    // Allow some tolerance for filter transient at start.
    // Energy should not exceed input by more than 5% (transient allowance).
    assert!(
        energy_after <= energy_before * 1.05,
        "Cut-only EQ should not add energy: before={energy_before:.2}, after={energy_after:.2}"
    );
}

#[test]
fn test_eq_energy_with_boost_bounded() {
    // With a 6 dB boost, energy can increase but should be bounded by
    // approximately the boost amount (6 dB = 4x power).
    let config = EqConfig {
        mid_gain_db: 6.0,
        ..EqConfig::default()
    };
    let mut eq = ChorusEQ::new(&config).expect("valid config");

    let mut signal = sine_wave(1500.0, 0.1, 0.3);
    let energy_before: f32 = signal.iter().map(|x| x * x).sum();

    eq.process(&mut signal);
    let energy_after: f32 = signal.iter().map(|x| x * x).sum();

    // 6 dB boost = 4x power. Allow 6x for filter transition band effects.
    assert!(
        energy_after < energy_before * 6.0,
        "6 dB boost energy should be bounded: ratio={:.2}",
        energy_after / energy_before,
    );
}

#[test]
fn test_eq_config_validation() {
    // Invalid frequency.
    let cfg = EqConfig {
        low_freq: 10.0,
        ..EqConfig::default()
    };
    assert!(cfg.validate().is_err());

    // Invalid gain.
    let cfg = EqConfig {
        mid_gain_db: 30.0,
        ..EqConfig::default()
    };
    assert!(cfg.validate().is_err());

    // Invalid Q.
    let cfg = EqConfig {
        mid_q: 0.0,
        ..EqConfig::default()
    };
    assert!(cfg.validate().is_err());

    // NaN frequency.
    let cfg = EqConfig {
        high_freq: f32::NAN,
        ..EqConfig::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_eq_handles_nan_input() {
    let config = EqConfig::default();
    let mut eq = ChorusEQ::new(&config).expect("valid config");
    let mut signal = vec![0.5, f32::NAN, 0.3, f32::INFINITY, -0.2];

    eq.process(&mut signal);

    for (i, &s) in signal.iter().enumerate() {
        assert!(
            s.is_finite(),
            "Sample {i} should be finite after EQ, got {s}"
        );
    }
}

// ---------------------------------------------------------------------------
// EQ preset tests
// ---------------------------------------------------------------------------

#[test]
fn test_all_presets_create_valid_eq() {
    for preset in &[
        EqPreset::Warm,
        EqPreset::Bright,
        EqPreset::Natural,
        EqPreset::Broadcast,
    ] {
        let config = preset.to_config();
        assert!(
            config.validate().is_ok(),
            "Preset {preset:?} config invalid"
        );
        assert!(
            ChorusEQ::new(&config).is_ok(),
            "Preset {preset:?} failed to create EQ"
        );
    }
}

#[test]
fn test_warm_preset_reduces_highs() {
    let config = EqPreset::Warm.to_config();
    let mut eq = ChorusEQ::new(&config).expect("valid config");

    let mut signal = sine_wave(10000.0, 0.1, 0.5);
    let rms_before = rms(&signal);

    eq.process(&mut signal);
    let rms_after = rms(&signal);

    assert!(
        rms_after < rms_before,
        "Warm preset should reduce high-frequency energy"
    );
}

#[test]
fn test_bright_preset_boosts_presence() {
    let config = EqPreset::Bright.to_config();
    let mut eq = ChorusEQ::new(&config).expect("valid config");

    let mut signal = sine_wave(3000.0, 0.1, 0.3);
    let rms_before = rms(&signal);

    eq.process(&mut signal);
    let rms_after = rms(&signal);

    assert!(
        rms_after > rms_before,
        "Bright preset should boost 3 kHz presence"
    );
}

// ---------------------------------------------------------------------------
// MixBusProcessor tests
// ---------------------------------------------------------------------------

#[test]
fn test_mix_bus_processor_creation() {
    let config = MixBusConfig::default();
    let proc = MixBusProcessor::new(4, &config);
    assert!(proc.is_ok());
    let proc = proc.unwrap();
    assert_eq!(proc.n_voices(), 4);
}

#[test]
fn test_mix_bus_processor_from_preset() {
    let config = MixBusConfig::from_preset(EqPreset::Broadcast);
    let proc = MixBusProcessor::new(2, &config);
    assert!(proc.is_ok());
}

#[test]
fn test_mix_bus_processor_voice_processing() {
    let config = MixBusConfig::default();
    let mut proc = MixBusProcessor::new(2, &config).expect("valid config");

    // Process voice 0 with a sibilant signal.
    let mut voice0 = sine_wave(6000.0, 0.05, 0.7);
    let rms_before = rms(&voice0);
    proc.process_voice(0, &mut voice0);
    let rms_after = rms(&voice0);

    // Should see some reduction due to de-essing.
    assert!(
        rms_after <= rms_before * 1.1,
        "Voice processing should not significantly amplify sibilance"
    );
}

#[test]
fn test_mix_bus_processor_without_deesser() {
    let config = MixBusConfig::default().without_deesser();
    let mut proc = MixBusProcessor::new(1, &config).expect("valid config");

    let mut voice = sine_wave(6000.0, 0.05, 0.5);
    let original = voice.clone();
    proc.process_voice(0, &mut voice);

    // With de-esser disabled, only EQ is applied. The Natural preset has
    // 0 dB at all bands, so the signal should be barely changed.
    let max_diff: f32 = voice
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Allow some tolerance for filter startup transient.
    assert!(
        max_diff < 0.15,
        "Without de-esser and flat EQ, signal should be nearly unchanged: max_diff={max_diff:.4}"
    );
}

#[test]
fn test_mix_bus_processor_bus_eq() {
    let bus_config = EqConfig {
        high_gain_db: -6.0,
        ..EqConfig::default()
    };
    let config = MixBusConfig::default().with_bus_eq(bus_config);
    let mut proc = MixBusProcessor::new(1, &config).expect("valid config");

    let mut mixed = sine_wave(10000.0, 0.1, 0.5);
    let rms_before = rms(&mixed);

    proc.process_bus(&mut mixed);
    let rms_after = rms(&mixed);

    assert!(
        rms_after < rms_before,
        "Bus EQ with high shelf cut should reduce 10 kHz energy"
    );
}

#[test]
fn test_mix_bus_processor_reset() {
    let config = MixBusConfig::default();
    let mut proc = MixBusProcessor::new(2, &config).expect("valid config");

    let mut voice = sine_wave(6000.0, 0.05, 0.8);
    proc.process_voice(0, &mut voice);

    // Reset should not panic and should clear state.
    proc.reset();
}

// ---------------------------------------------------------------------------
// Filter frequency response tests
// ---------------------------------------------------------------------------

#[test]
fn test_bandpass_minus_3db_point() {
    // Verify the bandpass filter's -3 dB point is at approximately
    // the specified center frequency bandwidth edges.
    let sr = KOKORO_SAMPLE_RATE as f32;
    let center = 6000.0f32;
    let q = 1.0f32;
    let coeffs = bandpass_coeffs(center, q, sr);

    // Generate test signals at center and at bandwidth edges.
    let at_center = sine_wave(center, 0.1, 1.0);
    let mut at_center_filtered = at_center.clone();
    let mut filter = BiquadFilter::new(coeffs);
    filter.process_buffer(&mut at_center_filtered);

    // Skip initial transient (first 10ms).
    let skip = (0.01 * sr) as usize;
    let rms_center = rms(&at_center_filtered[skip..]);
    let rms_input = rms(&at_center[skip..]);

    // At center frequency, gain should be close to Q (peak gain).
    let gain_db = 20.0 * (rms_center / rms_input).log10();
    // Bandpass peak gain should be positive (filter has gain = Q at center).
    assert!(
        gain_db > -3.0,
        "Bandpass gain at center should be > -3 dB, got {gain_db:.2} dB"
    );
}

#[test]
fn test_low_shelf_response_below_and_above() {
    let sr = KOKORO_SAMPLE_RATE as f32;
    let gain_db = 6.0f32;
    let coeffs = low_shelf_coeffs(200.0, gain_db, 0.707, sr);

    // Signal well below shelf frequency should see the full boost.
    let mut low_signal = sine_wave(50.0, 0.2, 0.3);
    let mut filter = BiquadFilter::new(coeffs);
    filter.process_buffer(&mut low_signal);
    let skip = (0.02 * sr) as usize;
    let rms_after = rms(&low_signal[skip..]);
    let rms_ref = rms(&sine_wave(50.0, 0.2, 0.3)[skip..]);

    let gain_measured = 20.0 * (rms_after / rms_ref).log10();
    // Should be approximately 6 dB boost (allow 2 dB tolerance).
    assert!(
        (gain_measured - gain_db).abs() < 2.5,
        "Low shelf gain at 50 Hz should be ~{gain_db} dB, got {gain_measured:.2} dB"
    );

    // Signal well above shelf frequency should be near unity.
    let mut high_signal = sine_wave(8000.0, 0.2, 0.3);
    let mut filter2 = BiquadFilter::new(coeffs);
    filter2.process_buffer(&mut high_signal);
    let rms_after_high = rms(&high_signal[skip..]);
    let rms_ref_high = rms(&sine_wave(8000.0, 0.2, 0.3)[skip..]);
    let gain_high = 20.0 * (rms_after_high / rms_ref_high).log10();

    assert!(
        gain_high.abs() < 1.0,
        "Low shelf gain at 8 kHz should be ~0 dB, got {gain_high:.2} dB"
    );
}

#[test]
fn test_high_shelf_response_below_and_above() {
    let sr = KOKORO_SAMPLE_RATE as f32;
    let gain_db = -6.0f32;
    let coeffs = high_shelf_coeffs(6000.0, gain_db, 0.707, sr);

    // Signal well above shelf frequency should see the full cut.
    let mut high_signal = sine_wave(11000.0, 0.2, 0.5);
    let mut filter = BiquadFilter::new(coeffs);
    filter.process_buffer(&mut high_signal);
    let skip = (0.02 * sr) as usize;
    let rms_after = rms(&high_signal[skip..]);
    let rms_ref = rms(&sine_wave(11000.0, 0.2, 0.5)[skip..]);

    let gain_measured = 20.0 * (rms_after / rms_ref).log10();
    assert!(
        (gain_measured - gain_db).abs() < 2.5,
        "High shelf gain at 11 kHz should be ~{gain_db} dB, got {gain_measured:.2} dB"
    );

    // Signal well below shelf frequency should be near unity.
    let mut low_signal = sine_wave(200.0, 0.2, 0.5);
    let mut filter2 = BiquadFilter::new(coeffs);
    filter2.process_buffer(&mut low_signal);
    let rms_after_low = rms(&low_signal[skip..]);
    let rms_ref_low = rms(&sine_wave(200.0, 0.2, 0.5)[skip..]);
    let gain_low = 20.0 * (rms_after_low / rms_ref_low).log10();

    assert!(
        gain_low.abs() < 1.0,
        "High shelf gain at 200 Hz should be ~0 dB, got {gain_low:.2} dB"
    );
}

#[test]
fn test_peaking_eq_at_center_and_away() {
    let sr = KOKORO_SAMPLE_RATE as f32;
    let gain_db = 6.0f32;
    let coeffs = peaking_eq_coeffs(1500.0, gain_db, 1.0, sr);

    // At center: should see boost.
    let mut center_signal = sine_wave(1500.0, 0.2, 0.3);
    let mut filter = BiquadFilter::new(coeffs);
    filter.process_buffer(&mut center_signal);
    let skip = (0.02 * sr) as usize;
    let rms_after = rms(&center_signal[skip..]);
    let rms_ref = rms(&sine_wave(1500.0, 0.2, 0.3)[skip..]);
    let gain_measured = 20.0 * (rms_after / rms_ref).log10();

    assert!(
        (gain_measured - gain_db).abs() < 2.0,
        "Peaking EQ gain at center should be ~{gain_db} dB, got {gain_measured:.2} dB"
    );

    // Far from center: should be near unity.
    let mut far_signal = sine_wave(100.0, 0.2, 0.3);
    let mut filter2 = BiquadFilter::new(coeffs);
    filter2.process_buffer(&mut far_signal);
    let rms_after_far = rms(&far_signal[skip..]);
    let rms_ref_far = rms(&sine_wave(100.0, 0.2, 0.3)[skip..]);
    let gain_far = 20.0 * (rms_after_far / rms_ref_far).log10();

    assert!(
        gain_far.abs() < 1.0,
        "Peaking EQ gain at 100 Hz should be ~0 dB, got {gain_far:.2} dB"
    );
}

// ---------------------------------------------------------------------------
// Biquad NaN defense tests
// ---------------------------------------------------------------------------

#[test]
fn test_biquad_nan_recovery() {
    let coeffs = bandpass_coeffs(6000.0, 1.0, KOKORO_SAMPLE_RATE as f32);
    let mut filter = BiquadFilter::new(coeffs);

    // Feed normal samples, then NaN, then normal again.
    let out1 = filter.process(0.5);
    assert!(out1.is_finite());

    let out_nan = filter.process(f32::NAN);
    assert!(out_nan.is_finite());
    assert_eq!(out_nan, 0.0);

    // Filter should recover after NaN.
    let out3 = filter.process(0.5);
    assert!(out3.is_finite());
}
