// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for [`super::kokoro_chorus_hrtf`].

use super::*;

#[test]
fn test_hrtf_config_default_valid() {
    let config = HrtfConfig::new().with_positions(semicircle(2));
    config.validate().expect("default config should be valid");
}

#[test]
fn test_hrtf_config_invalid_head_radius() {
    let config = HrtfConfig::new().with_head_radius_cm(-1.0);
    assert!(config.validate().is_err());
}

#[test]
fn test_hrtf_config_nan_speed_rejected() {
    let config = HrtfConfig::new().with_speed_of_sound(f32::NAN);
    assert!(config.validate().is_err());
}

#[test]
fn test_hrtf_config_invalid_position_distance() {
    let config = HrtfConfig::new().with_positions(vec![HrtfPosition::new(0.0, 0.0, 0.01)]);
    assert!(config.validate().is_err());
}

#[test]
fn test_semicircle_layout() {
    let positions = semicircle(5);
    assert_eq!(positions.len(), 5);
    // First should be at -90 deg, last at +90 deg
    assert!((positions[0].azimuth_deg - (-90.0)).abs() < 0.1);
    assert!((positions[4].azimuth_deg - 90.0).abs() < 0.1);
    // Middle should be near 0
    assert!((positions[2].azimuth_deg).abs() < 0.1);
}

#[test]
fn test_arc_layout() {
    let positions = arc(3, 60.0);
    assert_eq!(positions.len(), 3);
    assert!((positions[0].azimuth_deg - (-30.0)).abs() < 0.1);
    assert!((positions[1].azimuth_deg).abs() < 0.1);
    assert!((positions[2].azimuth_deg - 30.0).abs() < 0.1);
}

#[test]
fn test_surround_layout() {
    let positions = surround(4);
    assert_eq!(positions.len(), 4);
    // All at 2.0m distance
    for p in &positions {
        assert!((p.distance_m - 2.0).abs() < 0.01);
    }
}

#[test]
fn test_processor_creation() {
    let config = HrtfConfig::new().with_positions(semicircle(4));
    let proc = HrtfProcessor::new(&config, 24000.0).expect("processor creation");
    assert_eq!(proc.n_voices(), 4);
    assert!((proc.sample_rate() - 24000.0).abs() < 0.1);
}

#[test]
fn test_processor_invalid_sample_rate() {
    let config = HrtfConfig::new().with_positions(semicircle(2));
    assert!(HrtfProcessor::new(&config, 0.0).is_err());
    assert!(HrtfProcessor::new(&config, -1.0).is_err());
    assert!(HrtfProcessor::new(&config, f32::NAN).is_err());
}

#[test]
fn test_process_voices_silence() {
    let config = HrtfConfig::new().with_positions(semicircle(2));
    let mut proc = HrtfProcessor::new(&config, 24000.0).unwrap();
    let voices = vec![vec![0.0f32; 480], vec![0.0f32; 480]];
    let (left, right) = proc.process_voices(&voices).unwrap();
    assert_eq!(left.len(), 480);
    assert_eq!(right.len(), 480);
    for &s in &left {
        assert!(s.abs() < 1e-10);
    }
    for &s in &right {
        assert!(s.abs() < 1e-10);
    }
}

#[test]
fn test_process_voices_wrong_count() {
    let config = HrtfConfig::new().with_positions(semicircle(2));
    let mut proc = HrtfProcessor::new(&config, 24000.0).unwrap();
    let voices = vec![vec![1.0f32; 100]]; // 1 voice, expected 2
    assert!(proc.process_voices(&voices).is_err());
}

#[test]
fn test_itd_asymmetry() {
    // Source at +90 degrees (right): left ear should have more delay
    let config = HrtfConfig::new().with_positions(vec![HrtfPosition::new(90.0, 0.0, 1.5)]);
    let proc = HrtfProcessor::new(&config, 48000.0).unwrap();
    assert!(
        proc.itd_left(0) > proc.itd_right(0),
        "left ITD {} should exceed right ITD {} for source on the right",
        proc.itd_left(0),
        proc.itd_right(0),
    );
}

#[test]
fn test_itd_front_center_zero() {
    // Source at 0 degrees: both ears should have zero ITD
    let config = HrtfConfig::new().with_positions(vec![HrtfPosition::front(1.5)]);
    let proc = HrtfProcessor::new(&config, 48000.0).unwrap();
    assert!(proc.itd_left(0).abs() < 0.01);
    assert!(proc.itd_right(0).abs() < 0.01);
}

#[test]
fn test_woodworth_itd_magnitude() {
    // At 90 degrees, Woodworth ITD = (r/c)(pi/2 + 1) ~ 0.66 ms for r=0.0875
    // At 48 kHz: ~31.7 samples
    let config = HrtfConfig::new()
        .with_hrtf_model(HrtfModel::SphericalHead)
        .with_positions(vec![HrtfPosition::new(90.0, 0.0, 1.5)]);
    let proc = HrtfProcessor::new(&config, 48000.0).unwrap();
    let itd = proc.itd_left(0);
    // Expected: (0.0875/343) * (PI/2 + sin(PI/2)) * 48000
    let expected = (DEFAULT_HEAD_RADIUS_M / DEFAULT_SPEED_OF_SOUND)
        * (std::f32::consts::FRAC_PI_2 + 1.0)
        * 48000.0;
    assert!(
        (itd - expected).abs() < 1.0,
        "ITD {itd} should be close to expected {expected}",
    );
}

#[test]
fn test_stereo_lateralization() {
    // A voice on the right should produce more energy in the right channel
    let config = HrtfConfig::new().with_positions(vec![HrtfPosition::new(90.0, 0.0, 0.5)]);
    let mut proc = HrtfProcessor::new(&config, 24000.0).unwrap();
    let voices = vec![vec![1.0f32; 2000]];
    let (left, right) = proc.process_voices(&voices).unwrap();
    let energy_l: f32 = left.iter().map(|s| s * s).sum();
    let energy_r: f32 = right.iter().map(|s| s * s).sum();
    assert!(
        energy_r > energy_l,
        "right energy {energy_r} should exceed left energy {energy_l} for source at +90 deg",
    );
}

#[test]
fn test_nan_defense() {
    let config = HrtfConfig::new().with_positions(vec![HrtfPosition::front(1.5)]);
    let mut proc = HrtfProcessor::new(&config, 24000.0).unwrap();
    let voices = vec![vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0]];
    let (left, right) = proc.process_voices(&voices).unwrap();
    for &s in &left {
        assert!(s.is_finite(), "left sample {s} is not finite");
    }
    for &s in &right {
        assert!(s.is_finite(), "right sample {s} is not finite");
    }
}

#[test]
fn test_reset_clears_state() {
    let config = HrtfConfig::new().with_positions(vec![HrtfPosition::front(1.5)]);
    let mut proc = HrtfProcessor::new(&config, 24000.0).unwrap();
    let voices = vec![vec![1.0f32; 100]];
    let _ = proc.process_voices(&voices).unwrap();
    proc.reset();
    // After reset, processing silence should yield silence
    let voices = vec![vec![0.0f32; 100]];
    let (left, right) = proc.process_voices(&voices).unwrap();
    for &s in &left {
        assert!(s.abs() < 1e-6, "left {s} should be ~0 after reset");
    }
    for &s in &right {
        assert!(s.abs() < 1e-6, "right {s} should be ~0 after reset");
    }
}

#[test]
fn test_disabled_passthrough() {
    let config = HrtfConfig::new()
        .with_enabled(false)
        .with_positions(vec![HrtfPosition::front(1.5)]);
    let mut proc = HrtfProcessor::new(&config, 24000.0).unwrap();
    let voices = vec![vec![1.0f32; 100]];
    let (left, right) = proc.process_voices(&voices).unwrap();
    // When disabled, output is zeros (no processing)
    assert_eq!(left.len(), 100);
    assert_eq!(right.len(), 100);
}

#[test]
fn test_simple_delay_model() {
    let config = HrtfConfig::new()
        .with_hrtf_model(HrtfModel::SimpleDelay)
        .with_positions(vec![HrtfPosition::new(90.0, 0.0, 1.5)]);
    let proc = HrtfProcessor::new(&config, 48000.0).unwrap();
    // SimpleDelay uses sin(theta) only, so ITD should be smaller
    // than SphericalHead at 90 degrees.
    let simple_itd = proc.itd_left(0);

    let config2 = HrtfConfig::new()
        .with_hrtf_model(HrtfModel::SphericalHead)
        .with_positions(vec![HrtfPosition::new(90.0, 0.0, 1.5)]);
    let proc2 = HrtfProcessor::new(&config2, 48000.0).unwrap();
    let spherical_itd = proc2.itd_left(0);

    assert!(
        spherical_itd > simple_itd,
        "SphericalHead ITD {spherical_itd} should exceed SimpleDelay ITD {simple_itd}",
    );
}

#[test]
fn test_distance_attenuation() {
    // Close voice should be louder than far voice
    let config_close = HrtfConfig::new().with_positions(vec![HrtfPosition::front(0.5)]);
    let config_far = HrtfConfig::new().with_positions(vec![HrtfPosition::front(5.0)]);

    let mut proc_close = HrtfProcessor::new(&config_close, 24000.0).unwrap();
    let mut proc_far = HrtfProcessor::new(&config_far, 24000.0).unwrap();

    let voices = vec![vec![1.0f32; 2000]];
    let (close_l, _) = proc_close.process_voices(&voices).unwrap();
    let (far_l, _) = proc_far.process_voices(&voices).unwrap();

    let close_energy: f32 = close_l.iter().map(|s| s * s).sum();
    let far_energy: f32 = far_l.iter().map(|s| s * s).sum();
    assert!(
        close_energy > far_energy,
        "close energy {close_energy} should exceed far energy {far_energy}",
    );
}

#[test]
fn test_empty_positions() {
    let config = HrtfConfig::new(); // No positions
    let mut proc = HrtfProcessor::new(&config, 24000.0).unwrap();
    let voices: Vec<Vec<f32>> = Vec::new();
    let (left, right) = proc.process_voices(&voices).unwrap();
    assert!(left.is_empty());
    assert!(right.is_empty());
}
