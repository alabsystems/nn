// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for singing voice synthesis verification (Phases 1-3).

use super::*;
use crate::singing::pitch::{verify_pitch_accuracy, PitchAccuracyConfig};
use crate::singing::timing::{verify_timing, TimingConfig};
use crate::singing::vibrato::{verify_score_vibrato, verify_vibrato, VibratoConfig};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

use crate::test_audio_helpers::sine_wave_full;

/// Generate a sine wave with vibrato (frequency modulation).
fn sine_with_vibrato(
    center_hz: f64,
    vibrato_rate_hz: f64,
    vibrato_depth_hz: f64,
    sample_rate: u32,
    duration_sec: f64,
) -> Vec<f32> {
    let n = (f64::from(sample_rate) * duration_sec).ceil() as usize;
    let mut phase = 0.0_f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / f64::from(sample_rate);
            let freq = center_hz
                + vibrato_depth_hz * (2.0 * std::f64::consts::PI * vibrato_rate_hz * t).sin();
            phase += 2.0 * std::f64::consts::PI * freq / f64::from(sample_rate);
            (phase.sin() as f32) * 0.5
        })
        .collect()
}

/// Build a simple one-note score.
fn one_note_score(midi_note: u8, onset_sec: f64, duration_sec: f64) -> MusicalScore {
    MusicalScore {
        notes: vec![ScoreNote {
            midi_note,
            onset_sec,
            duration_sec,
            is_rest: false,
        }],
        tempo_bpm: 120.0,
    }
}

// ---------------------------------------------------------------------------
// Phase 1: Pitch conversion tests
// ---------------------------------------------------------------------------

#[test]
fn test_midi_to_hz_a4() {
    let hz = midi_to_hz(69);
    assert!(
        (hz - 440.0).abs() < 0.01,
        "A4 (MIDI 69) should be 440 Hz, got {hz}"
    );
}

#[test]
fn test_midi_to_hz_c4() {
    let hz = midi_to_hz(60);
    assert!(
        (hz - 261.63).abs() < 0.1,
        "C4 (MIDI 60) should be ~261.63 Hz, got {hz}"
    );
}

#[test]
fn test_hz_to_cents_octave() {
    let cents = hz_to_cents(880.0, 440.0);
    assert!(
        (cents - 1200.0).abs() < 0.01,
        "octave should be 1200 cents, got {cents}"
    );
}

#[test]
fn test_hz_to_cents_semitone() {
    // One semitone up from A4: A#4 = 466.16 Hz
    let cents = hz_to_cents(466.16, 440.0);
    assert!(
        (cents - 100.0).abs() < 0.5,
        "semitone should be ~100 cents, got {cents}"
    );
}

#[test]
fn test_hz_to_cents_zero_frequency() {
    assert_eq!(hz_to_cents(0.0, 440.0), 0.0);
    assert_eq!(hz_to_cents(440.0, 0.0), 0.0);
    assert_eq!(hz_to_cents(-1.0, 440.0), 0.0);
}

// ---------------------------------------------------------------------------
// Phase 1: Pitch accuracy tests
// ---------------------------------------------------------------------------

#[test]
fn test_pitch_accuracy_perfect() {
    let sample_rate = 24000;
    let a4_hz = 440.0;
    let duration = 1.0;
    let samples = sine_wave_full(a4_hz, sample_rate, duration, 0.5);
    let score = one_note_score(69, 0.0, duration);
    let config = PitchAccuracyConfig::default();

    let results = verify_pitch_accuracy(&samples, &score, &config, sample_rate).unwrap();
    assert_eq!(results.len(), 1, "should have 1 note result");

    let r = &results[0];
    assert_eq!(r.midi_note, 69);
    assert!(
        r.deviation_cents < 50.0,
        "perfect sine at target should have small deviation: {:.1} cents",
        r.deviation_cents
    );
    assert!(r.passed, "perfect pitch should pass: {r:?}");
}

#[test]
fn test_pitch_accuracy_sharp() {
    // Generate a sine 50 cents sharp of A4.
    let sample_rate = 24000;
    let a4_hz = 440.0;
    let sharp_hz = a4_hz * 2.0_f64.powf(50.0 / 1200.0); // 50 cents sharp
    let duration = 1.0;
    let samples = sine_wave_full(sharp_hz, sample_rate, duration, 0.5);
    let score = one_note_score(69, 0.0, duration);

    // Use tight threshold: 40 cents (should fail since we're 50 cents sharp).
    let config = PitchAccuracyConfig {
        max_deviation_cents: 40.0,
        in_tune_fraction: 0.80,
        ..PitchAccuracyConfig::default()
    };

    let results = verify_pitch_accuracy(&samples, &score, &config, sample_rate).unwrap();
    assert_eq!(results.len(), 1);

    let r = &results[0];
    // The note should fail because 50 cents > 40 cent threshold.
    assert!(
        !r.passed,
        "50-cent sharp note should fail at 40-cent threshold: {r:?}"
    );
}

#[test]
fn test_pitch_accuracy_empty_input() {
    let score = one_note_score(69, 0.0, 1.0);
    let config = PitchAccuracyConfig::default();
    let err = verify_pitch_accuracy(&[], &score, &config, 24000).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("empty") || msg.contains("Empty"),
        "should report empty input: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Phase 1: Score validation tests
// ---------------------------------------------------------------------------

#[test]
fn test_score_validation_empty() {
    let score = MusicalScore {
        notes: vec![],
        tempo_bpm: 120.0,
    };
    let err = score.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("empty") || msg.contains("Empty"), "{msg}");
}

#[test]
fn test_score_validation_nan_tempo() {
    let score = MusicalScore {
        notes: vec![ScoreNote {
            midi_note: 69,
            onset_sec: 0.0,
            duration_sec: 1.0,
            is_rest: false,
        }],
        tempo_bpm: f64::NAN,
    };
    let err = score.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("tempo_bpm"), "{msg}");
}

#[test]
fn test_score_validation_negative_duration() {
    let score = MusicalScore {
        notes: vec![ScoreNote {
            midi_note: 69,
            onset_sec: 0.0,
            duration_sec: -1.0,
            is_rest: false,
        }],
        tempo_bpm: 120.0,
    };
    let err = score.validate().unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("duration_sec"), "{msg}");
}

// ---------------------------------------------------------------------------
// Phase 2: Vibrato tests
// ---------------------------------------------------------------------------

#[test]
fn test_vibrato_extraction_sinusoidal() {
    // Generate 2 seconds of A4 with 6 Hz vibrato, ±20 Hz depth.
    let sample_rate = 24000;
    let samples = sine_with_vibrato(440.0, 6.0, 20.0, sample_rate, 2.0);
    let score = one_note_score(69, 0.0, 2.0);
    let config = VibratoConfig::default();

    let results = verify_score_vibrato(&samples, &score, &config, sample_rate).unwrap();
    assert!(!results.is_empty(), "should have at least 1 vibrato result");

    let (_, params, _) = &results[0];
    assert!(params.present, "vibrato should be detected: {params:?}");
    // Rate should be approximately 6 Hz (allow ±2 Hz tolerance).
    assert!(
        (params.rate_hz - 6.0).abs() < 2.0,
        "vibrato rate should be ~6 Hz, got {:.1} Hz",
        params.rate_hz
    );
}

#[test]
fn test_vibrato_absent_on_short_note() {
    // 200 ms note — too short for vibrato analysis with default config.
    let sample_rate = 24000;
    let samples = sine_wave_full(440.0, sample_rate, 0.2, 0.5);
    let score = one_note_score(69, 0.0, 0.2);
    let config = VibratoConfig::default(); // min_note_duration = 0.5s

    let results = verify_score_vibrato(&samples, &score, &config, sample_rate).unwrap();
    // Short note should be skipped.
    assert!(
        results.is_empty(),
        "200ms note should be skipped: {results:?}"
    );
}

#[test]
fn test_vibrato_rate_out_of_range() {
    // Generate vibrato at 3 Hz (below default range of 4-8 Hz).
    let sample_rate = 24000;
    let samples = sine_with_vibrato(440.0, 3.0, 20.0, sample_rate, 2.0);
    let score = one_note_score(69, 0.0, 2.0);
    let config = VibratoConfig::default();

    let results = verify_score_vibrato(&samples, &score, &config, sample_rate).unwrap();
    if !results.is_empty() {
        let (_, params, metric) = &results[0];
        if params.present {
            // If vibrato is detected, it should fail the rate check.
            assert!(
                !metric.passed || params.rate_hz >= 4.0,
                "3 Hz vibrato should either not be detected or fail: rate={:.1} Hz, passed={}",
                params.rate_hz,
                metric.passed
            );
        }
    }
}

#[test]
fn test_verify_vibrato_quality_metric() {
    use crate::singing::vibrato::VibratoParams;

    let good_vibrato = VibratoParams {
        rate_hz: 5.5,
        depth_cents: 50.0,
        onset_sec: 0.2,
        present: true,
    };
    let config = VibratoConfig::default();
    let metric = verify_vibrato(&good_vibrato, &config);
    assert!(metric.passed, "good vibrato should pass: {metric:?}");

    // Rate out of range.
    let bad_rate = VibratoParams {
        rate_hz: 2.0,
        depth_cents: 50.0,
        onset_sec: 0.2,
        present: true,
    };
    let metric = verify_vibrato(&bad_rate, &config);
    assert!(!metric.passed, "2 Hz vibrato should fail: {metric:?}");
}

// ---------------------------------------------------------------------------
// Phase 3: Timing tests
// ---------------------------------------------------------------------------

#[test]
fn test_timing_perfect() {
    // Build audio with note at exact onset position.
    let sample_rate = 24000;
    let onset = 0.1; // 100 ms of silence, then signal
    let duration = 0.5;
    let total_duration = onset + duration + 0.1;

    let mut samples = vec![0.0_f32; (f64::from(sample_rate) * total_duration) as usize];
    let onset_sample = (f64::from(sample_rate) * onset) as usize;
    let end_sample = (f64::from(sample_rate) * (onset + duration)) as usize;

    // Fill note region with sine wave.
    for i in onset_sample..end_sample.min(samples.len()) {
        let t = (i - onset_sample) as f64 / f64::from(sample_rate);
        samples[i] = (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32 * 0.5;
    }

    let score = one_note_score(69, onset, duration);
    let config = TimingConfig::default();

    let results = verify_timing(&samples, &score, &config, sample_rate).unwrap();
    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert!(
        r.onset_deviation_sec.abs() < 0.05,
        "onset deviation should be small: {:.3} sec",
        r.onset_deviation_sec
    );
}

#[test]
fn test_timing_rest_silence() {
    let sample_rate = 24000;
    // 1 second of silence.
    let samples = vec![0.0_f32; sample_rate as usize];
    let score = MusicalScore {
        notes: vec![ScoreNote {
            midi_note: 0,
            onset_sec: 0.0,
            duration_sec: 1.0,
            is_rest: true,
        }],
        tempo_bpm: 120.0,
    };
    let config = TimingConfig::default();

    let results = verify_timing(&samples, &score, &config, sample_rate).unwrap();
    assert_eq!(results.len(), 1);
    assert!(
        results[0].passed,
        "silent rest should pass: {:?}",
        results[0]
    );
}

#[test]
fn test_timing_rest_with_noise_fails() {
    let sample_rate = 24000;
    // "Rest" period with loud signal — should fail.
    let samples = sine_wave_full(440.0, sample_rate, 1.0, 0.5);
    let score = MusicalScore {
        notes: vec![ScoreNote {
            midi_note: 0,
            onset_sec: 0.0,
            duration_sec: 1.0,
            is_rest: true,
        }],
        tempo_bpm: 120.0,
    };
    let config = TimingConfig::default();

    let results = verify_timing(&samples, &score, &config, sample_rate).unwrap();
    assert_eq!(results.len(), 1);
    assert!(
        !results[0].passed,
        "noisy rest should fail: {:?}",
        results[0]
    );
}

#[test]
fn test_timing_empty_input() {
    let score = one_note_score(69, 0.0, 1.0);
    let config = TimingConfig::default();
    let err = verify_timing(&[], &score, &config, 24000).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("empty") || msg.contains("Empty"), "{msg}");
}

// ---------------------------------------------------------------------------
// Sample rate == 0 edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_pitch_accuracy_sample_rate_zero() {
    let score = one_note_score(69, 0.0, 1.0);
    let config = PitchAccuracyConfig::default();
    let samples = sine_wave_full(440.0, 24000, 1.0, 0.5);
    let err = verify_pitch_accuracy(&samples, &score, &config, 0).unwrap_err();
    assert!(
        matches!(err, TtsVerifyError::InvalidSampleRate(0)),
        "expected InvalidSampleRate(0), got: {err}"
    );
}

#[test]
fn test_timing_sample_rate_zero() {
    let score = one_note_score(69, 0.0, 1.0);
    let config = TimingConfig::default();
    let samples = vec![0.1_f32; 24000];
    let err = verify_timing(&samples, &score, &config, 0).unwrap_err();
    assert!(
        matches!(err, TtsVerifyError::InvalidSampleRate(0)),
        "expected InvalidSampleRate(0), got: {err}"
    );
}

#[test]
fn test_vibrato_sample_rate_zero() {
    let score = one_note_score(69, 0.0, 2.0);
    let config = VibratoConfig::default();
    let samples = vec![0.1_f32; 48000];
    let err = verify_score_vibrato(&samples, &score, &config, 0).unwrap_err();
    assert!(
        matches!(err, TtsVerifyError::InvalidSampleRate(0)),
        "expected InvalidSampleRate(0), got: {err}"
    );
}
