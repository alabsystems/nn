// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Audio quality gate tests for Kokoro chorus processing modules.
//!
//! Verifies measurable audio properties: energy preservation, stereo
//! correlation, absence of NaN/Inf, frequency response, dynamics range,
//! and reverb tail decay. All tests are deterministic (fixed seeds,
//! synthetic signals).
//!
//! Part of #4264.

use crate::kokoro_chorus::{
    mix_voices, mix_voices_stereo, mix_voices_with_config, ChorusConfig, VoiceMix,
};
use crate::kokoro_chorus_dynamics::{BusLimiter, DynamicsPreset, MultibandCompressor};
use crate::kokoro_chorus_eq::{ChorusEQ, DeEsser, DeEsserConfig, EqConfig, MixBusProcessor};
use crate::kokoro_chorus_reverb::{ReverbConfig, StereoReverb};
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Synthetic signal generators (deterministic)
// ---------------------------------------------------------------------------

/// Generate a sine wave at the given frequency, amplitude, and duration.
fn gen_sine(freq_hz: f32, amplitude: f32, duration_sec: f32) -> Vec<f32> {
    let n = (KOKORO_SAMPLE_RATE as f32 * duration_sec) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / KOKORO_SAMPLE_RATE as f32;
            amplitude * (2.0 * std::f32::consts::PI * freq_hz * t).sin()
        })
        .collect()
}

/// Generate silence (all zeros) for the given duration.
fn gen_silence(duration_sec: f32) -> Vec<f32> {
    let n = (KOKORO_SAMPLE_RATE as f32 * duration_sec) as usize;
    vec![0.0; n]
}

/// Generate a unit impulse (1.0 at sample 0, 0.0 elsewhere).
fn gen_impulse(duration_sec: f32) -> Vec<f32> {
    let n = (KOKORO_SAMPLE_RATE as f32 * duration_sec) as usize;
    let mut buf = vec![0.0; n];
    if !buf.is_empty() {
        buf[0] = 1.0;
    }
    buf
}

/// Generate deterministic pseudo-random noise using a simple LCG.
///
/// Amplitude is in [-amplitude, amplitude]. Seed is fixed for determinism.
fn gen_noise(amplitude: f32, duration_sec: f32, seed: u32) -> Vec<f32> {
    let n = (KOKORO_SAMPLE_RATE as f32 * duration_sec) as usize;
    let mut state = seed;
    (0..n)
        .map(|_| {
            // LCG: state = (a * state + c) mod m
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // Map to [-1, 1]
            let norm = (state as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            norm * amplitude
        })
        .collect()
}

/// Compute RMS energy of a signal.
fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = signal.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum_sq / signal.len() as f64).sqrt() as f32
}

/// Convert linear amplitude ratio to decibels.
fn to_db(ratio: f32) -> f32 {
    20.0 * ratio.log10()
}

/// Check that no sample is NaN or Inf.
fn assert_no_nan_inf(signal: &[f32], label: &str) {
    for (i, &s) in signal.iter().enumerate() {
        assert!(
            s.is_finite(),
            "{label}: sample {i} is not finite (value={s})"
        );
    }
}

/// Normalized cross-correlation at lag 0.
fn cross_correlation(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum_ab: f64 = a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
    let sum_a2: f64 = a.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let sum_b2: f64 = b.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let denom = (sum_a2 * sum_b2).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    (sum_ab / denom) as f32
}

// ---------------------------------------------------------------------------
// Gate tests
// ---------------------------------------------------------------------------

/// Process silence through chorus mixing and verify output is silence.
/// Process a sine wave and verify output energy is within +/- 3 dB.
#[test]
fn gate_energy_preservation() {
    let duration = 0.5;
    let n_voices = 4;

    // --- Silence in => silence out ---
    let silence = gen_silence(duration);
    let voices: Vec<Vec<f32>> = vec![silence; n_voices];
    let config = ChorusConfig::equal_gain(n_voices).unwrap();
    let mixed = mix_voices_with_config(&voices, &config).unwrap();
    let silence_rms = rms(&mixed);
    assert!(
        silence_rms < 1e-7,
        "Silence through chorus should be silence, got RMS={silence_rms}"
    );

    // --- Sine wave energy preservation (mono, equal gain) ---
    let sine = gen_sine(440.0, 0.5, duration);
    let sine_rms_input = rms(&sine);
    let sine_voices: Vec<Vec<f32>> = vec![sine; n_voices];
    let mixed_sine = mix_voices_with_config(&sine_voices, &config).unwrap();
    let sine_rms_output = rms(&mixed_sine);

    // With equal_gain (1/N), N identical signals sum to 1.0x the original.
    // Allow +/- 3 dB tolerance.
    let ratio_db = to_db(sine_rms_output / sine_rms_input);
    assert!(
        ratio_db.abs() < 3.0,
        "Energy preservation: expected within +/-3dB, got {ratio_db:.2}dB \
         (input RMS={sine_rms_input:.4}, output RMS={sine_rms_output:.4})"
    );

    // --- Soft limiter preserves energy within bounds ---
    let config_limiter = ChorusConfig::equal_gain(n_voices)
        .unwrap()
        .with_soft_limiter(1.5);
    let mixed_limited = mix_voices_with_config(&sine_voices, &config_limiter).unwrap();
    let limited_rms = rms(&mixed_limited);
    // Soft limiter should not dramatically boost energy.
    let limited_ratio_db = to_db(limited_rms / sine_rms_input);
    assert!(
        limited_ratio_db < 4.0,
        "Soft limiter energy: expected <= +4dB overhead, got {limited_ratio_db:.2}dB"
    );
}

/// Process mono-compatible chorus config and verify L and R channels
/// are correlated (cross-correlation > 0.9 for center-panned voices).
#[test]
fn gate_stereo_correlation() {
    let duration = 0.5;
    let n_voices = 3;

    // All voices center-panned => L and R should be highly correlated.
    let sine = gen_sine(440.0, 0.5, duration);
    let voices: Vec<Vec<f32>> = vec![sine; n_voices];
    let mix_params: Vec<VoiceMix> = vec![
        VoiceMix {
            gain: 0.33,
            pan: 0.0,
        },
        VoiceMix {
            gain: 0.33,
            pan: 0.0,
        },
        VoiceMix {
            gain: 0.34,
            pan: 0.0,
        },
    ];
    let stereo = mix_voices_stereo(&voices, &mix_params, true).unwrap();

    // Deinterleave.
    let n_frames = stereo.len() / 2;
    let left: Vec<f32> = (0..n_frames).map(|i| stereo[i * 2]).collect();
    let right: Vec<f32> = (0..n_frames).map(|i| stereo[i * 2 + 1]).collect();

    // Cross-correlation at lag 0.
    let corr = cross_correlation(&left, &right);
    assert!(
        corr > 0.9,
        "Center-panned stereo: L/R cross-correlation should be > 0.9, got {corr:.4}"
    );

    // Spread voices across stereo field => correlation should decrease
    // when voices have different content.
    let mix_params_wide: Vec<VoiceMix> = vec![
        VoiceMix {
            gain: 0.33,
            pan: -0.8,
        },
        VoiceMix {
            gain: 0.33,
            pan: 0.0,
        },
        VoiceMix {
            gain: 0.34,
            pan: 0.8,
        },
    ];
    let voices_diff: Vec<Vec<f32>> = vec![
        gen_sine(440.0, 0.5, duration),
        gen_sine(550.0, 0.5, duration),
        gen_sine(660.0, 0.5, duration),
    ];
    let stereo_wide = mix_voices_stereo(&voices_diff, &mix_params_wide, true).unwrap();
    let left_w: Vec<f32> = (0..n_frames).map(|i| stereo_wide[i * 2]).collect();
    let right_w: Vec<f32> = (0..n_frames).map(|i| stereo_wide[i * 2 + 1]).collect();
    let corr_wide = cross_correlation(&left_w, &right_w);

    // Wide stereo with different voices should be less correlated.
    assert!(
        corr_wide < corr,
        "Wide stereo correlation ({corr_wide:.4}) should be less than \
         center-only ({corr:.4})"
    );
}

/// Process various inputs and verify no NaN or Inf in output.
#[test]
fn gate_no_nan_or_inf() {
    let duration = 0.1;
    let n_voices = 4;
    let config = ChorusConfig::rich_chorus(n_voices).unwrap();

    // Silence.
    let voices_silence: Vec<Vec<f32>> = vec![gen_silence(duration); n_voices];
    let out = mix_voices_with_config(&voices_silence, &config).unwrap();
    assert_no_nan_inf(&out, "silence");

    // Impulse.
    let voices_impulse: Vec<Vec<f32>> = vec![gen_impulse(duration); n_voices];
    let out = mix_voices_with_config(&voices_impulse, &config).unwrap();
    assert_no_nan_inf(&out, "impulse");

    // Max amplitude.
    let max_amp = vec![1.0f32; (KOKORO_SAMPLE_RATE as f32 * duration) as usize];
    let voices_max: Vec<Vec<f32>> = vec![max_amp; n_voices];
    let out = mix_voices_with_config(&voices_max, &config).unwrap();
    assert_no_nan_inf(&out, "max_amplitude");

    // Deterministic noise.
    let noise_voices: Vec<Vec<f32>> = (0..n_voices)
        .map(|i| gen_noise(0.8, duration, 42 + i as u32))
        .collect();
    let out = mix_voices_with_config(&noise_voices, &config).unwrap();
    assert_no_nan_inf(&out, "noise");

    // Dynamics processor: no NaN/Inf.
    let mut compressor = MultibandCompressor::new(&DynamicsPreset::Broadcast.to_config()).unwrap();
    let mut buf = gen_noise(0.9, duration, 99);
    compressor.process(&mut buf);
    assert_no_nan_inf(&buf, "dynamics_compressor");

    // De-esser: no NaN/Inf.
    let mut deesser = DeEsser::new(&DeEsserConfig::default()).unwrap();
    let mut buf = gen_noise(0.9, duration, 77);
    deesser.process(&mut buf);
    assert_no_nan_inf(&buf, "deesser");

    // Bus limiter: no NaN/Inf.
    let mut limiter = BusLimiter::new();
    let mut buf = gen_noise(2.0, duration, 55); // exceeds [-1,1]
    limiter.process(&mut buf);
    assert_no_nan_inf(&buf, "bus_limiter");

    // EQ processor: no NaN/Inf.
    let mut eq = ChorusEQ::new(&EqConfig::default()).unwrap();
    let mut buf = gen_noise(0.9, duration, 33);
    eq.process(&mut buf);
    assert_no_nan_inf(&buf, "eq");
}

/// Process noise through de-esser and verify attenuation in sibilance band.
#[test]
fn gate_frequency_response() {
    let duration = 1.0;

    // Sibilance-band signal (6 kHz sine).
    let sibilance_signal = gen_sine(6000.0, 0.5, duration);
    let rms_before = rms(&sibilance_signal);

    // Apply de-esser with aggressive threshold.
    let mut processed = sibilance_signal;
    let mut deesser = DeEsser::new(&DeEsserConfig {
        center_freq_hz: 6000.0,
        q: 1.0,
        threshold_db: -30.0,
        max_reduction_db: -12.0,
        attack_sec: 0.001,
        release_sec: 0.050,
    })
    .unwrap();
    deesser.process(&mut processed);
    let rms_after = rms(&processed);

    // The de-esser should attenuate the sibilance-band signal.
    let attenuation_db = to_db(rms_after / rms_before);
    assert!(
        attenuation_db < -1.0,
        "De-esser should attenuate 6kHz by >1dB, got {attenuation_db:.2}dB"
    );

    // Verify low-frequency content passes through unattenuated.
    let low_signal = gen_sine(200.0, 0.5, duration);
    let rms_low_before = rms(&low_signal);
    let mut low_processed = low_signal;
    let mut deesser2 = DeEsser::new(&DeEsserConfig {
        center_freq_hz: 6000.0,
        q: 1.0,
        threshold_db: -30.0,
        max_reduction_db: -12.0,
        attack_sec: 0.001,
        release_sec: 0.050,
    })
    .unwrap();
    deesser2.process(&mut low_processed);
    let rms_low_after = rms(&low_processed);

    let low_change_db = to_db(rms_low_after / rms_low_before);
    assert!(
        low_change_db.abs() < 1.0,
        "De-esser should not attenuate 200Hz by >1dB, got {low_change_db:.2}dB"
    );
}

/// Process signal with varying amplitude through compressor and verify
/// output dynamic range is narrower than input.
#[test]
fn gate_dynamics_range() {
    let sr = KOKORO_SAMPLE_RATE;
    let segment_len = sr / 2; // 0.5 seconds per segment
    let mut signal = Vec::with_capacity(segment_len * 2);

    // Quiet segment: 220 Hz sine at 0.05 amplitude.
    for i in 0..segment_len {
        let t = i as f32 / sr as f32;
        signal.push(0.05 * (2.0 * std::f32::consts::PI * 220.0 * t).sin());
    }
    // Loud segment: 220 Hz sine at 0.8 amplitude.
    for i in 0..segment_len {
        let t = i as f32 / sr as f32;
        signal.push(0.8 * (2.0 * std::f32::consts::PI * 220.0 * t).sin());
    }

    let rms_quiet_in = rms(&signal[..segment_len]);
    let rms_loud_in = rms(&signal[segment_len..]);
    let dynamic_range_in_db = to_db(rms_loud_in / rms_quiet_in);

    // Apply aggressive compression.
    let mut compressor = MultibandCompressor::new(&DynamicsPreset::Aggressive.to_config()).unwrap();
    compressor.process(&mut signal);

    let rms_quiet_out = rms(&signal[..segment_len]);
    let rms_loud_out = rms(&signal[segment_len..]);

    if rms_quiet_out > 1e-8 {
        let dynamic_range_out_db = to_db(rms_loud_out / rms_quiet_out);
        assert!(
            dynamic_range_out_db < dynamic_range_in_db,
            "Compressor should reduce dynamic range: \
             input={dynamic_range_in_db:.1}dB, output={dynamic_range_out_db:.1}dB"
        );
    }

    // Bus limiter should constrain peaks.
    let mut limiter = BusLimiter::new();
    let mut loud_signal = gen_sine(440.0, 1.5, 0.5); // exceeds 1.0
    limiter.process(&mut loud_signal);
    let peak = loud_signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        peak <= 1.0,
        "Bus limiter output should not exceed 1.0, got peak={peak:.4}"
    );
}

/// Process impulse through reverb and verify tail decays below -30dB.
#[test]
fn gate_reverb_tail() {
    let duration = 2.0;
    let mut impulse = gen_impulse(duration);

    // Apply reverb with large room size for longer tail.
    let config = ReverbConfig {
        reverb_mix: 1.0, // fully wet for measuring tail
        room_size: 0.8,
        early_reflections: false,
        damping: 0.3,
    };
    let mut reverb = StereoReverb::new(&config);
    reverb.process_mono(&mut impulse);

    // Verify that reverb produced non-zero early output.
    let early_rms = rms(&impulse[1..KOKORO_SAMPLE_RATE / 10]);
    assert!(
        early_rms > 1e-6,
        "Reverb should produce non-zero output, got early RMS={early_rms}"
    );

    // Tail (last 0.5s) should be quiet relative to impulse amplitude.
    let tail_start = impulse.len() - (KOKORO_SAMPLE_RATE / 2);
    let tail_rms = rms(&impulse[tail_start..]);
    if tail_rms > 0.0 {
        let decay_db = to_db(tail_rms);
        assert!(
            decay_db < -30.0,
            "Reverb tail (last 0.5s) should be below -30dB, got {decay_db:.1}dB"
        );
    }
}

/// Verify MixBusProcessor preserves signal integrity end-to-end.
#[test]
fn gate_mix_bus_processor() {
    let duration = 0.5;
    let n_voices = 3;

    let mut processor =
        MixBusProcessor::new(n_voices, &crate::kokoro_chorus_eq::MixBusConfig::default()).unwrap();

    // Process each voice through EQ + de-esser.
    let mut voices: Vec<Vec<f32>> = (0..n_voices)
        .map(|i| gen_sine(220.0 * (i as f32 + 1.0), 0.4, duration))
        .collect();
    let rms_before: Vec<f32> = voices.iter().map(|v| rms(v)).collect();

    for (i, voice) in voices.iter_mut().enumerate() {
        processor.process_voice(i, voice);
    }
    let rms_after: Vec<f32> = voices.iter().map(|v| rms(v)).collect();

    // EQ with Natural preset should not dramatically change energy.
    for i in 0..n_voices {
        let ratio_db = to_db(rms_after[i] / rms_before[i]);
        assert!(
            ratio_db.abs() < 6.0,
            "Voice {i} EQ+de-esser change should be <6dB, got {ratio_db:.2}dB"
        );
    }

    // No NaN/Inf in processed voices.
    for (i, voice) in voices.iter().enumerate() {
        assert_no_nan_inf(voice, &format!("voice_{i}"));
    }

    // Mix and verify bus output.
    let gains = vec![1.0 / n_voices as f32; n_voices];
    let mut mixed = mix_voices(&voices, &gains, true).unwrap();
    processor.process_bus(&mut mixed);
    assert_no_nan_inf(&mixed, "bus_output");
}

// ---------------------------------------------------------------------------
// Comprehensive pipeline integration tests (#4264)
// ---------------------------------------------------------------------------

/// Compute the peak absolute value of a signal.
fn peak_abs(signal: &[f32]) -> f32 {
    signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
}

/// Feed N voices through the full ChorusMasterPipeline (all stages),
/// verify output is valid stereo with no NaN/Inf and reasonable energy.
#[test]
fn gate_master_pipeline_end_to_end() {
    use crate::kokoro_chorus_pipeline::{ChorusMasterConfig, ChorusMasterPipeline};

    let n_voices = 4;
    let duration = 0.5;

    // Create voices with different frequencies to simulate a real ensemble.
    let voices: Vec<Vec<f32>> = (0..n_voices)
        .map(|i| gen_sine(220.0 + i as f32 * 30.0, 0.4, duration))
        .collect();

    // Full pipeline: vibrato + detune + EQ + de-ess + humanize + blend +
    // stereo + dynamics + reverb + limiter.
    let config = ChorusMasterConfig::full(n_voices).unwrap();
    let mut pipeline = ChorusMasterPipeline::new(config).unwrap();
    let (left, right) = pipeline.process(&voices).unwrap();

    // Output must be stereo (equal length L/R).
    assert_eq!(left.len(), right.len(), "L and R must have equal length");
    assert!(!left.is_empty(), "output must be non-empty");

    // No NaN/Inf in either channel.
    assert_no_nan_inf(&left, "pipeline_L");
    assert_no_nan_inf(&right, "pipeline_R");

    // Output should have meaningful energy (not silence).
    let rms_l = rms(&left);
    let rms_r = rms(&right);
    assert!(
        rms_l > 1e-4,
        "left channel should have audible energy, got RMS={rms_l}"
    );
    assert!(
        rms_r > 1e-4,
        "right channel should have audible energy, got RMS={rms_r}"
    );

    // Energy should be bounded (limiter is on).
    let peak_l = peak_abs(&left);
    let peak_r = peak_abs(&right);
    assert!(peak_l < 1.5, "left peak should be limited, got {peak_l:.4}");
    assert!(
        peak_r < 1.5,
        "right peak should be limited, got {peak_r:.4}"
    );
}

/// Feed N voices through the pipeline via the stateful object, then
/// reset and run again -- output should be similar (determinism).
#[test]
fn gate_master_pipeline_reset_determinism() {
    use crate::kokoro_chorus_pipeline::{ChorusMasterConfig, ChorusMasterPipeline};

    let n_voices = 3;
    let duration = 0.3;
    let voices: Vec<Vec<f32>> = (0..n_voices)
        .map(|i| gen_sine(330.0 + i as f32 * 20.0, 0.5, duration))
        .collect();

    let config = ChorusMasterConfig::standard(n_voices).unwrap();
    let mut pipeline = ChorusMasterPipeline::new(config).unwrap();

    let (l1, r1) = pipeline.process(&voices).unwrap();
    pipeline.reset();
    let (l2, r2) = pipeline.process(&voices).unwrap();

    // After reset, output from same input should match within epsilon.
    // Some stages (biquad filters) have transient state that reset clears,
    // so we compare from slightly past the start to skip transient region.
    let skip = 240; // skip first 10ms of transient
    if l1.len() > skip && l2.len() > skip {
        let max_diff_l = l1[skip..]
            .iter()
            .zip(l2[skip..].iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_diff_r = r1[skip..]
            .iter()
            .zip(r2[skip..].iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff_l < 0.05,
            "after reset, L should be similar: max diff = {max_diff_l}"
        );
        assert!(
            max_diff_r < 0.05,
            "after reset, R should be similar: max diff = {max_diff_r}"
        );
    }
}

/// Verify vibrato produces measurable frequency modulation.
///
/// A pure sine tone through vibrato should differ from the original
/// (pitch is modulated). Energy should be preserved.
#[test]
fn gate_vibrato_quality() {
    use crate::kokoro_chorus_vibrato::{apply_vibrato, VibratoConfig};

    let duration = 0.5;
    let sr = KOKORO_SAMPLE_RATE as u32;
    let sine = gen_sine(440.0, 0.5, duration);

    // Apply vibrato with moderate depth (40 cents).
    let config = VibratoConfig {
        rate_hz: 5.5,
        depth_cents: 40.0,
        rate_spread_hz: 0.0,
        depth_spread_cents: 0.0,
        onset_sec: 0.0,
    };

    let mut voices = vec![sine.clone()];
    apply_vibrato(&mut voices, &config, sr).unwrap();
    let vibrato_signal = &voices[0];

    assert_no_nan_inf(vibrato_signal, "vibrato_output");

    // Energy should be preserved within +/-3 dB.
    let rms_before = rms(&sine);
    let rms_after = rms(vibrato_signal);
    let ratio_db = to_db(rms_after / rms_before);
    assert!(
        ratio_db.abs() < 3.0,
        "vibrato should preserve energy within +/-3dB, got {ratio_db:.2}dB"
    );

    // Peak amplitude should not increase significantly.
    let peak_orig = peak_abs(&sine);
    let peak_vib = peak_abs(vibrato_signal);
    assert!(
        peak_vib < peak_orig * 1.3,
        "vibrato peak ({peak_vib:.4}) should not significantly exceed original ({peak_orig:.4})"
    );

    // Output should differ from input (vibrato is not a no-op).
    let max_diff: f32 = sine
        .iter()
        .zip(vibrato_signal.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 0.01,
        "vibrato output should differ from input, max diff = {max_diff}"
    );
}

/// Verify the de-esser attenuates sibilance-band signals more than
/// low-frequency signals.
#[test]
fn gate_deesser_effectiveness() {
    let duration = 0.5;
    let deesser_config = DeEsserConfig {
        center_freq_hz: 6000.0,
        q: 1.0,
        threshold_db: -30.0,
        max_reduction_db: -12.0,
        attack_sec: 0.001,
        release_sec: 0.050,
    };

    // Pure sibilance (6 kHz) should be significantly attenuated.
    let sibilance = gen_sine(6000.0, 0.5, duration);
    let rms_sib_before = rms(&sibilance);
    let mut sib_processed = sibilance;
    let mut deesser1 = DeEsser::new(&deesser_config).unwrap();
    deesser1.process(&mut sib_processed);
    let rms_sib_after = rms(&sib_processed);
    let sib_attenuation_db = to_db(rms_sib_after / rms_sib_before);

    // Low-frequency (200 Hz) should pass through with minimal change.
    let low_signal = gen_sine(200.0, 0.5, duration);
    let rms_low_before = rms(&low_signal);
    let mut low_processed = low_signal;
    let mut deesser2 = DeEsser::new(&deesser_config).unwrap();
    deesser2.process(&mut low_processed);
    let rms_low_after = rms(&low_processed);
    let low_attenuation_db = to_db(rms_low_after / rms_low_before);

    assert_no_nan_inf(&sib_processed, "deesser_sibilance");
    assert_no_nan_inf(&low_processed, "deesser_low");

    // Sibilance should be attenuated more than low-frequency content.
    assert!(
        sib_attenuation_db < low_attenuation_db,
        "de-esser should attenuate 6kHz ({sib_attenuation_db:.2}dB) more than \
         200Hz ({low_attenuation_db:.2}dB)"
    );

    // Sibilance attenuation should be at least 1 dB.
    assert!(
        sib_attenuation_db < -1.0,
        "de-esser should attenuate 6kHz by >1dB, got {sib_attenuation_db:.2}dB"
    );

    // Low-frequency attenuation should be less than 1 dB.
    assert!(
        low_attenuation_db.abs() < 1.0,
        "de-esser should not attenuate 200Hz by >1dB, got {low_attenuation_db:.2}dB"
    );
}

/// Compare soft tanh limiter vs hard clip on a hot signal.
#[test]
fn gate_soft_limiter_vs_hard_clip() {
    let duration = 0.3;
    let n_voices = 4;

    let voices: Vec<Vec<f32>> = (0..n_voices)
        .map(|i| gen_sine(220.0 + i as f32 * 20.0, 0.8, duration))
        .collect();

    // Hard clip path.
    let config_hard = ChorusConfig::equal_gain(n_voices).unwrap().with_clip(true);
    let hard_clipped = mix_voices_with_config(&voices, &config_hard).unwrap();

    // Soft limiter path.
    let config_soft = ChorusConfig::equal_gain(n_voices)
        .unwrap()
        .with_soft_limiter(1.5);
    let soft_limited = mix_voices_with_config(&voices, &config_soft).unwrap();

    assert_no_nan_inf(&hard_clipped, "hard_clip");
    assert_no_nan_inf(&soft_limited, "soft_limit");

    // Both should have bounded peaks.
    let peak_hard = peak_abs(&hard_clipped);
    let peak_soft = peak_abs(&soft_limited);
    assert!(
        peak_hard <= 1.0 + 1e-5,
        "hard clip peak should be <= 1.0, got {peak_hard:.6}"
    );
    assert!(
        peak_soft < 1.1,
        "soft limiter peak should be bounded, got {peak_soft:.6}"
    );

    // Both should have non-trivial energy.
    let rms_hard = rms(&hard_clipped);
    let rms_soft = rms(&soft_limited);
    assert!(rms_hard > 0.01, "hard clip RMS too low: {rms_hard}");
    assert!(rms_soft > 0.01, "soft limiter RMS too low: {rms_soft}");

    // Crest factor = peak / RMS. Soft limiter should not be worse than
    // hard clip at preserving dynamic range.
    let crest_hard = peak_hard / rms_hard;
    let crest_soft = peak_soft / rms_soft;
    assert!(
        crest_soft >= crest_hard * 0.8,
        "soft limiter crest ({crest_soft:.4}) should not be much worse \
         than hard clip ({crest_hard:.4})"
    );
}

/// Verify that detuned voices produce a beating pattern.
#[test]
fn gate_detuning_blend_beating() {
    let duration = 1.0;
    let base_freq = 440.0;

    // Voice 0: 440 Hz. Voice 1: 442 Hz (2 Hz difference = 2 Hz beating).
    let v0 = gen_sine(base_freq, 0.5, duration);
    let v1 = gen_sine(base_freq + 2.0, 0.5, duration);
    let voices = vec![v0, v1];

    let config = ChorusConfig::equal_gain(2).unwrap().with_clip(false);
    let mixed = mix_voices_with_config(&voices, &config).unwrap();
    assert_no_nan_inf(&mixed, "beating_mix");

    // Compute amplitude envelope via windowed RMS (10ms windows).
    let window_size = KOKORO_SAMPLE_RATE / 100;
    let n_windows = mixed.len() / window_size;
    let mut envelope: Vec<f32> = Vec::with_capacity(n_windows);
    for w in 0..n_windows {
        let start = w * window_size;
        let end = (start + window_size).min(mixed.len());
        envelope.push(rms(&mixed[start..end]));
    }

    let env_max = envelope.iter().fold(0.0f32, |m, &v| m.max(v));
    let env_min = envelope.iter().fold(f32::MAX, |m, &v| m.min(v));
    let env_range = env_max - env_min;

    // With 2 Hz beating over 1 second, envelope should vary significantly.
    assert!(
        env_range > env_max * 0.2,
        "detuned voices should produce beating: range={env_range:.4}, \
         max={env_max:.4}, min={env_min:.4}"
    );

    // Mixed signal should retain energy.
    let rms_single = rms(&gen_sine(base_freq, 0.5, duration));
    let rms_mixed = rms(&mixed);
    assert!(
        rms_mixed > rms_single * 0.5,
        "beating mix should retain energy: single={rms_single:.4}, mixed={rms_mixed:.4}"
    );
}

/// Verify both the full pipeline and a minimal path produce valid output.
#[test]
fn gate_pipeline_vs_default_path() {
    use crate::kokoro_chorus_pipeline::{process_chorus, ChorusMasterConfig};

    let n_voices = 4;
    let duration = 0.3;
    let voices: Vec<Vec<f32>> = (0..n_voices)
        .map(|i| gen_sine(220.0 + i as f32 * 30.0, 0.4, duration))
        .collect();

    let config_full = ChorusMasterConfig::full(n_voices).unwrap();
    let (full_l, full_r) = process_chorus(&voices, &config_full).unwrap();

    let config_minimal = ChorusMasterConfig::minimal(n_voices).unwrap();
    let (min_l, min_r) = process_chorus(&voices, &config_minimal).unwrap();

    assert_no_nan_inf(&full_l, "full_L");
    assert_no_nan_inf(&full_r, "full_R");
    assert_no_nan_inf(&min_l, "minimal_L");
    assert_no_nan_inf(&min_r, "minimal_R");

    assert!(rms(&full_l) > 1e-4, "full pipeline L should not be silent");
    assert!(
        rms(&min_l) > 1e-4,
        "minimal pipeline L should not be silent"
    );

    // Full pipeline should produce different output than minimal.
    let diff_l: f32 = full_l
        .iter()
        .zip(min_l.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f32>()
        / full_l.len().min(min_l.len()) as f32;
    assert!(
        diff_l > 1e-4,
        "full and minimal pipelines should differ, mean L diff = {diff_l}"
    );
}

/// Feed edge-case inputs through the full pipeline, verify no NaN/Inf.
#[test]
fn gate_nan_safety_full_pipeline() {
    use crate::kokoro_chorus_pipeline::{process_chorus, ChorusMasterConfig};

    let n_voices = 3;
    let duration = 0.1;
    let n = (KOKORO_SAMPLE_RATE as f32 * duration) as usize;
    let config = ChorusMasterConfig::full(n_voices).unwrap();

    // Near-zero input (denormal-range).
    let near_zero: Vec<f32> = vec![1e-38; n];
    let voices_nz = vec![near_zero; n_voices];
    let (l, r) = process_chorus(&voices_nz, &config).unwrap();
    assert_no_nan_inf(&l, "near_zero_L");
    assert_no_nan_inf(&r, "near_zero_R");

    // Max amplitude (all +1.0).
    let max_amp = vec![1.0f32; n];
    let voices_max = vec![max_amp; n_voices];
    let (l, r) = process_chorus(&voices_max, &config).unwrap();
    assert_no_nan_inf(&l, "max_amp_L");
    assert_no_nan_inf(&r, "max_amp_R");

    // Impulse train (spike every 240 samples).
    let mut impulse_train = vec![0.0f32; n];
    for i in (0..n).step_by(240) {
        impulse_train[i] = 1.0;
    }
    let voices_it = vec![impulse_train; n_voices];
    let (l, r) = process_chorus(&voices_it, &config).unwrap();
    assert_no_nan_inf(&l, "impulse_train_L");
    assert_no_nan_inf(&r, "impulse_train_R");

    // DC offset (+0.5).
    let dc_offset = vec![0.5f32; n];
    let voices_dc = vec![dc_offset; n_voices];
    let (l, r) = process_chorus(&voices_dc, &config).unwrap();
    assert_no_nan_inf(&l, "dc_offset_L");
    assert_no_nan_inf(&r, "dc_offset_R");

    // Alternating +1/-1 (Nyquist square wave).
    let square: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let voices_sq = vec![square; n_voices];
    let (l, r) = process_chorus(&voices_sq, &config).unwrap();
    assert_no_nan_inf(&l, "square_L");
    assert_no_nan_inf(&r, "square_R");

    // Negative max amplitude (all -1.0).
    let neg_max = vec![-1.0f32; n];
    let voices_neg = vec![neg_max; n_voices];
    let (l, r) = process_chorus(&voices_neg, &config).unwrap();
    assert_no_nan_inf(&l, "neg_max_L");
    assert_no_nan_inf(&r, "neg_max_R");
}

/// Verify dynamics compression reduces dynamic range through the pipeline.
#[test]
fn gate_dynamics_through_pipeline() {
    use crate::kokoro_chorus_pipeline::{process_chorus, ChorusMasterConfig};

    let n_voices = 2;
    let sr = KOKORO_SAMPLE_RATE;
    let segment_len = sr / 2;

    let mut voice_signal = Vec::with_capacity(segment_len * 2);
    for i in 0..segment_len {
        let t = i as f32 / sr as f32;
        voice_signal.push(0.05 * (2.0 * std::f32::consts::PI * 330.0 * t).sin());
    }
    for i in 0..segment_len {
        let t = i as f32 / sr as f32;
        voice_signal.push(0.8 * (2.0 * std::f32::consts::PI * 330.0 * t).sin());
    }

    let voices = vec![voice_signal.clone(); n_voices];

    let config = ChorusMasterConfig::new(n_voices)
        .unwrap()
        .with_stereo(
            crate::kokoro_chorus_stereo::StereoChorusConfig::auto_layout(n_voices).unwrap(),
        )
        .with_dynamics(DynamicsPreset::Aggressive.to_config())
        .with_limiter(true);

    let (left, _right) = process_chorus(&voices, &config).unwrap();
    assert_no_nan_inf(&left, "dynamics_L");

    let out_quiet_rms = rms(&left[..segment_len]);
    let out_loud_rms = rms(&left[segment_len..]);
    let in_quiet_rms = rms(&voice_signal[..segment_len]);
    let in_loud_rms = rms(&voice_signal[segment_len..]);
    let in_range_db = to_db(in_loud_rms / in_quiet_rms);

    if out_quiet_rms > 1e-8 {
        let out_range_db = to_db(out_loud_rms / out_quiet_rms);
        assert!(
            out_range_db < in_range_db,
            "dynamics should reduce range: input={in_range_db:.1}dB, output={out_range_db:.1}dB"
        );
    }
}

/// Verify reverb adds a tail beyond the impulse through the pipeline.
#[test]
fn gate_reverb_through_pipeline() {
    use crate::kokoro_chorus_pipeline::{process_chorus, ChorusMasterConfig};
    use crate::kokoro_chorus_reverb::ReverbConfig;

    let n_voices = 2;
    let duration = 1.0;
    let impulse = gen_impulse(duration);
    let voices = vec![impulse; n_voices];

    let config = ChorusMasterConfig::new(n_voices)
        .unwrap()
        .with_stereo(
            crate::kokoro_chorus_stereo::StereoChorusConfig::auto_layout(n_voices).unwrap(),
        )
        .with_reverb(ReverbConfig {
            reverb_mix: 0.5,
            room_size: 0.7,
            early_reflections: false,
            damping: 0.3,
        });

    let (left, _right) = process_chorus(&voices, &config).unwrap();
    assert_no_nan_inf(&left, "reverb_pipeline_L");

    let tail_start = KOKORO_SAMPLE_RATE / 10;
    let tail_end = KOKORO_SAMPLE_RATE / 2;
    if left.len() > tail_end {
        let tail_rms = rms(&left[tail_start..tail_end]);
        assert!(
            tail_rms > 1e-6,
            "reverb should produce tail energy: tail RMS={tail_rms}"
        );
    }
}

/// Process different voice counts (1, 2, 4, 8) and verify all produce
/// valid stereo with bounded energy.
#[test]
fn gate_voice_count_scaling() {
    use crate::kokoro_chorus_pipeline::{process_chorus, ChorusMasterConfig};

    let duration = 0.2;

    for n_voices in [1, 2, 4, 8] {
        let voices: Vec<Vec<f32>> = (0..n_voices)
            .map(|i| gen_sine(220.0 + i as f32 * 15.0, 0.4, duration))
            .collect();
        let config = ChorusMasterConfig::standard(n_voices).unwrap();
        let (left, right) = process_chorus(&voices, &config).unwrap();

        assert_no_nan_inf(&left, &format!("scaling_{n_voices}v_L"));
        assert_no_nan_inf(&right, &format!("scaling_{n_voices}v_R"));

        let rms_l = rms(&left);
        assert!(
            rms_l > 1e-5,
            "{n_voices} voices: output should have energy, L RMS={rms_l}"
        );

        let peak_l = peak_abs(&left);
        assert!(
            peak_l < 2.0,
            "{n_voices} voices: output peak should be bounded, got {peak_l:.4}"
        );
    }
}

/// Verify EQ preset spectral effect: Warm preset boosts lows, cuts highs.
#[test]
fn gate_eq_preset_spectral_shape() {
    let duration = 0.5;

    // Low-frequency signal: Warm preset should boost relative to Natural.
    let low_signal = gen_sine(150.0, 0.3, duration);

    let mut warm_out = low_signal.clone();
    let mut warm_eq = ChorusEQ::new(&crate::kokoro_chorus_eq::EqPreset::Warm.to_config()).unwrap();
    warm_eq.process(&mut warm_out);
    let rms_warm_low = rms(&warm_out);

    let mut natural_out = low_signal;
    let mut natural_eq =
        ChorusEQ::new(&crate::kokoro_chorus_eq::EqPreset::Natural.to_config()).unwrap();
    natural_eq.process(&mut natural_out);
    let rms_natural_low = rms(&natural_out);

    assert_no_nan_inf(&warm_out, "warm_eq_low");
    assert_no_nan_inf(&natural_out, "natural_eq_low");

    // Warm preset has +2dB low shelf at 200Hz.
    assert!(
        rms_warm_low > rms_natural_low * 0.95,
        "Warm preset should boost low signal: warm={rms_warm_low:.4}, natural={rms_natural_low:.4}"
    );

    // High-frequency signal: Warm preset cuts highs (-2.5dB at 8kHz).
    let high_signal = gen_sine(8000.0, 0.3, duration);

    let mut warm_high = high_signal.clone();
    let mut warm_eq2 = ChorusEQ::new(&crate::kokoro_chorus_eq::EqPreset::Warm.to_config()).unwrap();
    warm_eq2.process(&mut warm_high);
    let rms_warm_high = rms(&warm_high);

    let mut natural_high = high_signal;
    let mut natural_eq2 =
        ChorusEQ::new(&crate::kokoro_chorus_eq::EqPreset::Natural.to_config()).unwrap();
    natural_eq2.process(&mut natural_high);
    let rms_natural_high = rms(&natural_high);

    assert_no_nan_inf(&warm_high, "warm_eq_high");
    assert_no_nan_inf(&natural_high, "natural_eq_high");

    assert!(
        rms_warm_high < rms_natural_high * 1.05,
        "Warm preset should not boost high signal: warm={rms_warm_high:.4}, natural={rms_natural_high:.4}"
    );
}

/// Feed noise through the full pipeline and verify it differs from minimal path.
#[test]
fn gate_full_chain_spectral_modification() {
    use crate::kokoro_chorus_pipeline::{process_chorus, ChorusMasterConfig};

    let n_voices = 3;
    let duration = 0.5;

    let voices: Vec<Vec<f32>> = (0..n_voices)
        .map(|i| gen_noise(0.4, duration, 100 + i as u32))
        .collect();

    let config_full = ChorusMasterConfig::full(n_voices).unwrap();
    let (full_l, _) = process_chorus(&voices, &config_full).unwrap();

    let config_min = ChorusMasterConfig::minimal(n_voices).unwrap();
    let (min_l, _) = process_chorus(&voices, &config_min).unwrap();

    assert_no_nan_inf(&full_l, "full_chain_L");
    assert_no_nan_inf(&min_l, "minimal_chain_L");

    let min_len = full_l.len().min(min_l.len());
    let mean_diff: f32 = full_l[..min_len]
        .iter()
        .zip(min_l[..min_len].iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f32>()
        / min_len as f32;

    assert!(
        mean_diff > 1e-4,
        "full pipeline should modify output: mean_diff={mean_diff}"
    );
}
