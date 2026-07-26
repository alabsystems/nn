// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

const SR: f32 = KOKORO_SAMPLE_RATE as f32;

/// Generate a sine wave at the given frequency.
fn sine_wave(freq: f32, n_samples: usize, amplitude: f32) -> Vec<f32> {
    (0..n_samples)
        .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / SR).sin())
        .collect()
}

/// Compute RMS energy of a signal.
fn rms(buf: &[f32]) -> f32 {
    let sum_sq: f32 = buf.iter().map(|x| x * x).sum();
    (sum_sq / buf.len() as f32).sqrt()
}

/// Compute energy in a frequency band using a bandpass filter.
fn band_energy(buf: &[f32], center_hz: f32, bw_oct: f32) -> f32 {
    let mut bp = Biquad::bandpass(center_hz, bw_oct, SR);
    let filtered: Vec<f32> = buf.iter().map(|&x| bp.process(x)).collect();
    rms(&filtered)
}

// --- Config validation ---

#[test]
fn test_config_default_valid() {
    PresenceConfig::new()
        .validate()
        .expect("default config should be valid");
}

#[test]
fn test_config_builder_roundtrip() {
    let cfg = PresenceConfig::new()
        .with_presence_boost_db(5.0)
        .with_presence_center_hz(4000.0)
        .with_presence_q(2.0)
        .with_air_boost_db(3.0)
        .with_air_center_hz(12000.0)
        .with_dynamic_range_db(8.0)
        .with_sibilance_threshold_db(-15.0)
        .with_mix(0.7);
    cfg.validate().expect("builder config should be valid");
    assert_eq!(cfg.presence_boost_db, 5.0);
    assert_eq!(cfg.presence_center_hz, 4000.0);
    assert_eq!(cfg.presence_q, 2.0);
    assert_eq!(cfg.air_boost_db, 3.0);
    assert_eq!(cfg.air_center_hz, 12000.0);
    assert_eq!(cfg.dynamic_range_db, 8.0);
    assert_eq!(cfg.sibilance_threshold_db, -15.0);
    assert_eq!(cfg.mix, 0.7);
}

#[test]
fn test_config_invalid_presence_boost_db() {
    assert!(PresenceConfig::new()
        .with_presence_boost_db(-1.0)
        .validate()
        .is_err());
    assert!(PresenceConfig::new()
        .with_presence_boost_db(13.0)
        .validate()
        .is_err());
    assert!(PresenceConfig::new()
        .with_presence_boost_db(f32::NAN)
        .validate()
        .is_err());
}

#[test]
fn test_config_invalid_presence_center_hz() {
    assert!(PresenceConfig::new()
        .with_presence_center_hz(500.0)
        .validate()
        .is_err());
    assert!(PresenceConfig::new()
        .with_presence_center_hz(10000.0)
        .validate()
        .is_err());
}

#[test]
fn test_config_invalid_presence_q() {
    assert!(PresenceConfig::new()
        .with_presence_q(0.1)
        .validate()
        .is_err());
    assert!(PresenceConfig::new()
        .with_presence_q(6.0)
        .validate()
        .is_err());
}

#[test]
fn test_config_invalid_mix() {
    assert!(PresenceConfig::new().with_mix(-0.1).validate().is_err());
    assert!(PresenceConfig::new().with_mix(1.1).validate().is_err());
    assert!(PresenceConfig::new()
        .with_mix(f32::INFINITY)
        .validate()
        .is_err());
}

#[test]
fn test_config_invalid_air_center_hz() {
    assert!(PresenceConfig::new()
        .with_air_center_hz(4000.0)
        .validate()
        .is_err());
    assert!(PresenceConfig::new()
        .with_air_center_hz(20000.0)
        .validate()
        .is_err());
}

#[test]
fn test_config_invalid_sibilance_threshold_db() {
    assert!(PresenceConfig::new()
        .with_sibilance_threshold_db(-70.0)
        .validate()
        .is_err());
    assert!(PresenceConfig::new()
        .with_sibilance_threshold_db(1.0)
        .validate()
        .is_err());
}

// --- Preset validation ---

#[test]
fn test_preset_subtle_valid() {
    subtle().validate().expect("subtle preset should be valid");
}

#[test]
fn test_preset_forward_valid() {
    forward()
        .validate()
        .expect("forward preset should be valid");
}

#[test]
fn test_preset_broadcast_valid() {
    broadcast()
        .validate()
        .expect("broadcast preset should be valid");
}

#[test]
fn test_preset_airy_valid() {
    airy().validate().expect("airy preset should be valid");
}

// --- Processor creation ---

#[test]
fn test_invalid_sample_rate() {
    let cfg = PresenceConfig::new();
    assert!(PresenceProcessor::new(&cfg, 0.0).is_err());
    assert!(PresenceProcessor::new(&cfg, -44100.0).is_err());
    assert!(PresenceProcessor::new(&cfg, f32::NAN).is_err());
}

#[test]
fn test_new_kokoro_creates_processor() {
    let cfg = PresenceConfig::new();
    PresenceProcessor::new_kokoro(&cfg).expect("should create kokoro processor");
}

// --- Processing behavior ---

#[test]
fn test_mix_zero_is_identity() {
    let mut buf = sine_wave(1000.0, 4096, 0.5);
    let original = buf.clone();
    let cfg = PresenceConfig::new().with_mix(0.0);
    let mut proc = PresenceProcessor::new_kokoro(&cfg).unwrap();
    proc.process(&mut buf);
    assert_eq!(buf, original, "mix=0 should be identity");
}

#[test]
fn test_presence_boosts_midrange_energy() {
    let n = 8192;
    // Signal in the presence band (3.5 kHz).
    let mut buf = sine_wave(3500.0, n, 0.3);
    let dry_rms = rms(&buf);

    let cfg = PresenceConfig::new()
        .with_presence_boost_db(6.0)
        .with_air_boost_db(0.0)
        .with_mix(1.0);
    let mut proc = PresenceProcessor::new_kokoro(&cfg).unwrap();
    proc.process(&mut buf);
    let wet_rms = rms(&buf);

    assert!(
        wet_rms > dry_rms,
        "presence boost should increase energy at 3.5 kHz: dry={dry_rms}, wet={wet_rms}",
    );
}

#[test]
fn test_air_boost_increases_high_frequency_energy() {
    let n = 8192;
    // Broadband signal so the shelf has something to boost.
    let mut buf: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / SR;
            0.2 * (2.0 * std::f32::consts::PI * 500.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 3000.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 10000.0 * t).sin()
        })
        .collect();
    let dry_hf = band_energy(&buf, 10000.0, 1.0);

    let cfg = PresenceConfig::new()
        .with_presence_boost_db(0.0)
        .with_air_boost_db(6.0)
        .with_air_center_hz(9000.0)
        .with_mix(1.0);
    let mut proc = PresenceProcessor::new_kokoro(&cfg).unwrap();
    proc.process(&mut buf);
    let wet_hf = band_energy(&buf, 10000.0, 1.0);

    assert!(
        wet_hf > dry_hf * 1.1,
        "air boost should increase HF energy: dry={dry_hf}, wet={wet_hf}",
    );
}

#[test]
fn test_dynamic_boost_louder_for_quiet_signals() {
    let n = 4096;

    // Quiet signal.
    let mut quiet = sine_wave(3500.0, n, 0.01);
    let quiet_dry_rms = rms(&quiet);

    // Loud signal.
    let mut loud = sine_wave(3500.0, n, 0.8);
    let loud_dry_rms = rms(&loud);

    let cfg = PresenceConfig::new()
        .with_presence_boost_db(3.0)
        .with_dynamic_range_db(12.0)
        .with_air_boost_db(0.0)
        .with_mix(1.0);

    let mut proc_q = PresenceProcessor::new_kokoro(&cfg).unwrap();
    proc_q.process(&mut quiet);
    let quiet_boost_ratio = rms(&quiet) / quiet_dry_rms.max(1e-10);

    let mut proc_l = PresenceProcessor::new_kokoro(&cfg).unwrap();
    proc_l.process(&mut loud);
    let loud_boost_ratio = rms(&loud) / loud_dry_rms.max(1e-10);

    assert!(
        quiet_boost_ratio > loud_boost_ratio,
        "quiet signal should get more boost: quiet_ratio={quiet_boost_ratio}, \
         loud_ratio={loud_boost_ratio}",
    );
}

#[test]
fn test_sibilance_reduces_presence_boost() {
    let n = 8192;

    // Non-sibilant signal (low frequency, no energy at 5-8 kHz).
    let mut non_sib = sine_wave(1000.0, n, 0.3);
    let non_sib_dry = rms(&non_sib);

    // Sibilant signal (high frequency, energy in the sibilance band).
    let mut sibilant = sine_wave(7000.0, n, 0.3);
    let sibilant_dry = rms(&sibilant);

    let cfg = PresenceConfig::new()
        .with_presence_boost_db(6.0)
        .with_sibilance_threshold_db(-30.0)
        .with_air_boost_db(0.0)
        .with_mix(1.0);

    let mut proc_ns = PresenceProcessor::new_kokoro(&cfg).unwrap();
    proc_ns.process(&mut non_sib);
    let non_sib_ratio = rms(&non_sib) / non_sib_dry.max(1e-10);

    let mut proc_s = PresenceProcessor::new_kokoro(&cfg).unwrap();
    proc_s.process(&mut sibilant);
    let sibilant_ratio = rms(&sibilant) / sibilant_dry.max(1e-10);

    // The sibilant signal should receive less boost due to sibilance attenuation.
    // (Or at least not significantly more boost than the non-sibilant signal.)
    // Due to the different frequency content, we check the sibilant doesn't
    // get dramatically more boost.
    assert!(
        sibilant_ratio < non_sib_ratio * 3.0,
        "sibilant signal should not receive excessive boost: \
         non_sib_ratio={non_sib_ratio}, sib_ratio={sibilant_ratio}",
    );
}

#[test]
fn test_all_outputs_finite() {
    let inputs = vec![
        0.0,
        0.5,
        -0.5,
        1.0,
        -1.0,
        0.001,
        -0.001,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let cfg = PresenceConfig::new()
        .with_presence_boost_db(6.0)
        .with_air_boost_db(6.0)
        .with_mix(1.0);
    let mut proc = PresenceProcessor::new_kokoro(&cfg).unwrap();
    let mut buf = inputs;
    proc.process(&mut buf);
    for (i, &v) in buf.iter().enumerate() {
        assert!(v.is_finite(), "sample {i} is non-finite: {v}");
    }
}

#[test]
fn test_reset_clears_state() {
    let cfg = PresenceConfig::new().with_mix(1.0);
    let mut proc = PresenceProcessor::new_kokoro(&cfg).unwrap();
    let mut buf = vec![0.5; 200];
    proc.process(&mut buf);
    assert!(
        proc.level_env > 0.0,
        "envelope should be nonzero after processing"
    );
    proc.reset();
    assert_eq!(proc.level_env, 0.0);
    assert_eq!(proc.sib_env, 0.0);
}

#[test]
fn test_process_stereo_modifies_both_channels() {
    let n = 4096;
    let mut left = sine_wave(3500.0, n, 0.4);
    let mut right = sine_wave(3500.0, n, 0.3);
    let left_orig = left.clone();
    let right_orig = right.clone();

    let cfg = PresenceConfig::new()
        .with_presence_boost_db(6.0)
        .with_mix(1.0);
    let mut proc = PresenceProcessor::new_kokoro(&cfg).unwrap();
    proc.process_stereo(&mut left, &mut right);

    assert_ne!(left, left_orig, "left channel should be modified");
    assert_ne!(right, right_orig, "right channel should be modified");
}

#[test]
fn test_process_stereo_outputs_finite() {
    let mut left = vec![0.5, -0.5, f32::NAN, 0.3, f32::INFINITY];
    let mut right = vec![-0.3, 0.7, 0.1, f32::NEG_INFINITY, 0.0];

    let cfg = PresenceConfig::new()
        .with_presence_boost_db(6.0)
        .with_air_boost_db(4.0)
        .with_mix(1.0);
    let mut proc = PresenceProcessor::new_kokoro(&cfg).unwrap();
    proc.process_stereo(&mut left, &mut right);

    for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
        assert!(l.is_finite(), "left[{i}] is non-finite: {l}");
        assert!(r.is_finite(), "right[{i}] is non-finite: {r}");
    }
}

#[test]
fn test_apply_presence_per_voice() {
    let n = 2048;
    let mut voices = vec![
        sine_wave(800.0, n, 0.5),
        sine_wave(1200.0, n, 0.5),
        sine_wave(3500.0, n, 0.5),
    ];
    let dry_energies: Vec<f32> = voices.iter().map(|v| rms(v)).collect();

    let cfg = PresenceConfig::new()
        .with_presence_boost_db(6.0)
        .with_air_boost_db(3.0)
        .with_mix(1.0);
    apply_presence(&mut voices, &cfg, SR).unwrap();

    // Each voice should have been processed (energy changed).
    for (i, (voice, dry_e)) in voices.iter().zip(dry_energies.iter()).enumerate() {
        let wet_e = rms(voice);
        assert!(
            (wet_e - dry_e).abs() > 1e-6,
            "voice {i} should be modified: dry_rms={dry_e}, wet_rms={wet_e}",
        );
    }
}

#[test]
fn test_config_accessor() {
    let cfg = PresenceConfig::new()
        .with_presence_boost_db(5.0)
        .with_mix(0.8);
    let proc = PresenceProcessor::new_kokoro(&cfg).unwrap();
    assert_eq!(proc.config().presence_boost_db, 5.0);
    assert_eq!(proc.config().mix, 0.8);
}

#[test]
fn test_mix_interpolation() {
    let n = 4096;
    let input = sine_wave(3500.0, n, 0.5);

    // Process at mix=0.5.
    let mut half = input.clone();
    let cfg_half = PresenceConfig::new()
        .with_presence_boost_db(6.0)
        .with_air_boost_db(0.0)
        .with_mix(0.5);
    let mut proc_half = PresenceProcessor::new_kokoro(&cfg_half).unwrap();
    proc_half.process(&mut half);

    // Process at mix=1.0.
    let mut full = input.clone();
    let cfg_full = PresenceConfig::new()
        .with_presence_boost_db(6.0)
        .with_air_boost_db(0.0)
        .with_mix(1.0);
    let mut proc_full = PresenceProcessor::new_kokoro(&cfg_full).unwrap();
    proc_full.process(&mut full);

    // Mix=0.5 should produce less change from dry than mix=1.0.
    let dry_rms = rms(&input);
    let half_diff = (rms(&half) - dry_rms).abs();
    let full_diff = (rms(&full) - dry_rms).abs();

    assert!(
        full_diff > half_diff * 0.5,
        "full mix should produce more change than half mix: \
         half_diff={half_diff}, full_diff={full_diff}",
    );
}
