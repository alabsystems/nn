// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive production integration tests for the Kokoro chorus pipeline.
//!
//! Exercises the complete processing chain end-to-end: full pipeline smoke,
//! preset round-trips, signal chain ordering, saturation coloration, spatial
//! positioning, alignment tightening, formant preservation, breath insertion,
//! NaN safety, and determinism.
//!
//! Part of #4264.

use crate::kokoro_chorus_alignment::{align_voices, cross_correlate, AlignmentConfig};
use crate::kokoro_chorus_breath::{
    detect_pauses, insert_breath_sounds, BreathConfig, BreathGenerator,
};
use crate::kokoro_chorus_formant::{shift_pitch_preserve_formant, simple_pitch_shift};
use crate::kokoro_chorus_pipeline::{process_chorus, ChorusMasterConfig, ChorusMasterPipeline};
use crate::kokoro_chorus_preset_library::ChorusPreset;
use crate::kokoro_chorus_saturation::{SaturationConfig, SaturationMode, SaturationProcessor};
use crate::kokoro_chorus_spatial::{
    auto_layout_spatial, process_voice_spatial, SpatialConfig, VoiceSpatialPosition,
};
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a mono sine wave at the given frequency, duration, and amplitude.
fn make_sine(freq_hz: f32, duration_sec: f32, amplitude: f32) -> Vec<f32> {
    let n = (KOKORO_SAMPLE_RATE as f32 * duration_sec) as usize;
    (0..n)
        .map(|i| {
            amplitude
                * (2.0 * std::f32::consts::PI * freq_hz * i as f32 / KOKORO_SAMPLE_RATE as f32)
                    .sin()
        })
        .collect()
}

/// Generate synthetic voices: N sine waves at slightly different frequencies.
fn make_voices(n: usize, base_freq: f32, duration_sec: f32) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| make_sine(base_freq + i as f32 * 15.0, duration_sec, 0.5))
        .collect()
}

/// RMS energy of a signal.
fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let energy: f32 = signal.iter().map(|s| s * s).sum();
    (energy / signal.len() as f32).sqrt()
}

/// Total energy of a signal (sum of squared samples).
fn total_energy(signal: &[f32]) -> f64 {
    signal.iter().map(|&s| f64::from(s) * f64::from(s)).sum()
}

// ===========================================================================
// 1. Full pipeline smoke test
// ===========================================================================

#[test]
fn test_production_full_pipeline_smoke() {
    let voices = make_voices(4, 220.0, 0.5);
    let config = ChorusMasterConfig::full(4).expect("full config");
    let mut pipeline = ChorusMasterPipeline::new(config).expect("pipeline construction");
    let (left, right) = pipeline.process(&voices).expect("pipeline process");

    // Output must be stereo with matching lengths.
    assert_eq!(left.len(), right.len(), "L and R length mismatch");
    assert_eq!(left.len(), voices[0].len(), "output length != input length");

    // All samples must be finite.
    for (i, &s) in left.iter().enumerate() {
        assert!(s.is_finite(), "left[{i}] is not finite: {s}");
    }
    for (i, &s) in right.iter().enumerate() {
        assert!(s.is_finite(), "right[{i}] is not finite: {s}");
    }

    // Stereo channels should differ (voices are panned).
    let diff: f32 = left
        .iter()
        .zip(right.iter())
        .map(|(&l, &r)| (l - r).abs())
        .sum::<f32>()
        / left.len() as f32;
    assert!(
        diff > 1e-6,
        "stereo channels should differ, mean diff = {diff}"
    );
}

// ===========================================================================
// 2. Preset round-trip — all ChorusPreset variants
// ===========================================================================

#[test]
fn test_production_preset_round_trip() {
    let n_voices = 4;
    let voices = make_voices(n_voices, 300.0, 0.3);

    for &preset in ChorusPreset::ALL {
        let config = preset
            .to_config(n_voices)
            .unwrap_or_else(|e| panic!("{preset:?} to_config failed: {e}"));
        config
            .validate()
            .unwrap_or_else(|e| panic!("{preset:?} validate failed: {e}"));

        let (left, right) = process_chorus(&voices, &config)
            .unwrap_or_else(|e| panic!("{preset:?} process failed: {e}"));

        assert_eq!(left.len(), right.len(), "{preset:?}: L/R length mismatch");
        for &s in &left {
            assert!(s.is_finite(), "{preset:?}: left has non-finite sample");
        }
        for &s in &right {
            assert!(s.is_finite(), "{preset:?}: right has non-finite sample");
        }
    }
}

// ===========================================================================
// 3. Signal chain ordering
// ===========================================================================

#[test]
fn test_production_signal_chain_ordering() {
    // A pipeline with only stereo should differ from one with EQ + stereo,
    // demonstrating that per-voice EQ affects the downstream stereo mix.
    let voices = make_voices(4, 440.0, 0.3);

    let stereo_only = ChorusMasterConfig::minimal(4).expect("minimal");
    let with_eq = ChorusMasterConfig::standard(4).expect("standard");

    let (l_stereo, _) = process_chorus(&voices, &stereo_only).expect("stereo-only");
    let (l_eq, _) = process_chorus(&voices, &with_eq).expect("with-eq");

    let max_diff: f32 = l_stereo
        .iter()
        .zip(l_eq.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-4,
        "EQ should change the signal, max_diff = {max_diff}"
    );
}

// ===========================================================================
// 4. Saturation coloration
// ===========================================================================

#[test]
fn test_production_saturation_coloration() {
    // Process a clean sine through tape saturation at high drive.
    // The waveshaping introduces harmonics, changing the signal's spectral
    // content. We verify that the output differs from the input (energy
    // changes or shape changes).
    let sine = make_sine(440.0, 0.2, 0.5);
    let dry_energy = total_energy(&sine);

    let cfg = SaturationConfig::new()
        .with_drive(0.8)
        .with_mix(1.0)
        .with_mode(SaturationMode::Tape)
        .with_output_gain_db(0.0);
    let mut processor = SaturationProcessor::new_kokoro(cfg).expect("valid saturation config");

    let mut saturated = sine.clone();
    processor.process(&mut saturated);

    let wet_energy = total_energy(&saturated);

    // Energy should change measurably (harmonics redistribute energy).
    let energy_ratio = (wet_energy - dry_energy).abs() / dry_energy.max(1e-30);
    assert!(
        energy_ratio > 0.001,
        "tape saturation should change energy: dry={dry_energy:.4}, wet={wet_energy:.4}, ratio={energy_ratio:.6}",
    );

    // Output must still be finite.
    for (i, &s) in saturated.iter().enumerate() {
        assert!(s.is_finite(), "saturated[{i}] is not finite: {s}");
    }

    // Verify all four saturation modes produce different outputs.
    let modes = [
        SaturationMode::Tape,
        SaturationMode::Tube,
        SaturationMode::Console,
        SaturationMode::Warm,
    ];
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for mode in modes {
        let mode_cfg = SaturationConfig::new()
            .with_drive(0.6)
            .with_mix(1.0)
            .with_mode(mode)
            .with_output_gain_db(0.0);
        let mut proc = SaturationProcessor::new_kokoro(mode_cfg).expect("valid");
        let mut buf = sine.clone();
        proc.process(&mut buf);
        outputs.push(buf);
    }
    // Each mode should produce a measurably different output.
    for i in 0..modes.len() {
        for j in (i + 1)..modes.len() {
            let diff: f32 = outputs[i]
                .iter()
                .zip(outputs[j].iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / outputs[i].len() as f32;
            assert!(
                diff > 1e-5,
                "{:?} vs {:?} should produce different output, mean_diff = {diff}",
                modes[i],
                modes[j],
            );
        }
    }
}

// ===========================================================================
// 5. Spatial positioning — closer voices are louder
// ===========================================================================

#[test]
fn test_production_spatial_positioning() {
    let config = SpatialConfig::new();

    // A voice at 0.5m (near) should be louder than one at 6.0m (far).
    let near_pos = VoiceSpatialPosition::new(0.5, 0.0, 0.0);
    let far_pos = VoiceSpatialPosition::new(6.0, 0.0, 0.0);

    // Use a longer signal to ensure the propagation delay flushes.
    let mono = make_sine(440.0, 0.3, 0.8);

    let (near_l, near_r) =
        process_voice_spatial(&mono, &config, &near_pos).expect("near voice spatial");
    let (far_l, far_r) =
        process_voice_spatial(&mono, &config, &far_pos).expect("far voice spatial");

    let near_energy = total_energy(&near_l) + total_energy(&near_r);
    let far_energy = total_energy(&far_l) + total_energy(&far_r);

    assert!(
        near_energy > far_energy,
        "near voice ({near_energy:.2}) should have more energy than far voice ({far_energy:.2})"
    );

    // Additionally verify auto layout produces valid positions.
    let positions = auto_layout_spatial(6, &config).expect("auto layout for 6 voices");
    assert_eq!(positions.len(), 6);
    for pos in &positions {
        assert!(pos.distance >= 0.1, "distance must be >= MIN_DISTANCE");
        assert!(pos.distance <= config.room_size);
        assert!(pos.angle.is_finite());
        assert!(pos.elevation.is_finite());
    }
}

// ===========================================================================
// 6. Alignment tightening
// ===========================================================================

#[test]
fn test_production_alignment_tightening() {
    // Create a reference signal and a copy delayed by 3 samples.
    let n = 4096;
    let freq = 5.0;
    let reference: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / n as f32).sin())
        .collect();

    // Shift the target by 3 samples.
    let shift = 3;
    let mut target = vec![0.0f32; n];
    target[shift..n].copy_from_slice(&reference[..n - shift]);

    // Measure cross-correlation before alignment.
    let pre_corr = cross_correlate(&reference, &target, 10).expect("pre-alignment cross-correlate");

    // Align voices.
    let config = AlignmentConfig::new(0.8)
        .expect("valid alignment config")
        .with_max_shift(20)
        .with_correlation_window(1024);
    let voices = vec![reference.clone(), target.clone()];
    let aligned = align_voices(&voices, &config).expect("align voices");

    // Measure cross-correlation after alignment.
    let post_corr =
        cross_correlate(&reference, &aligned[1], 10).expect("post-alignment cross-correlate");

    assert!(
        post_corr.coefficient >= pre_corr.coefficient - 0.01,
        "alignment should improve or maintain correlation: pre={:.4}, post={:.4}",
        pre_corr.coefficient,
        post_corr.coefficient,
    );

    // The aligned lag should be closer to 0 than the original lag.
    assert!(
        post_corr.lag.unsigned_abs() <= pre_corr.lag.unsigned_abs(),
        "alignment should reduce lag: pre_lag={}, post_lag={}",
        pre_corr.lag,
        post_corr.lag,
    );
}

// ===========================================================================
// 7. Formant preservation
// ===========================================================================

#[test]
fn test_production_formant_preservation() {
    // Generate a signal with formant-like spectral structure (two harmonics).
    let n = 8192;
    let sr = KOKORO_SAMPLE_RATE as f32;
    let audio: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr;
            // Fundamental at 200 Hz + formant-like harmonic at 800 Hz.
            0.5 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 800.0 * t).sin()
        })
        .collect();

    // Shift pitch up by 2 semitones.
    let ratio = 2.0f32.powf(2.0 / 12.0);

    // Formant-preserving shift.
    let formant_shifted = shift_pitch_preserve_formant(&audio, ratio, None).expect("formant shift");

    // Simple (non-preserving) shift for comparison.
    let simple_shifted = simple_pitch_shift(&audio, ratio);

    // Both must have same length and be finite.
    assert_eq!(formant_shifted.len(), audio.len());
    assert_eq!(simple_shifted.len(), audio.len());
    for (i, &s) in formant_shifted.iter().enumerate() {
        assert!(s.is_finite(), "formant_shifted[{i}] not finite: {s}");
    }

    // The formant-preserving shift should differ from the simple shift
    // (the formant compensation changes the spectral balance).
    let diff: f32 = formant_shifted
        .iter()
        .zip(simple_shifted.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / n as f32;
    assert!(
        diff > 1e-4,
        "formant-preserving shift should differ from simple shift, mean_diff = {diff}"
    );

    // Measure energy in the formant region (500-1200 Hz bin range).
    // The formant-preserving version should preserve more energy in this
    // region relative to total energy than the simple version.
    let formant_energy_fp = band_energy(&formant_shifted, sr, 500.0, 1200.0);
    let total_energy_fp = total_energy(&formant_shifted);
    let formant_ratio_fp = formant_energy_fp / total_energy_fp.max(1e-30);

    let formant_energy_simple = band_energy(&simple_shifted, sr, 500.0, 1200.0);
    let total_energy_simple = total_energy(&simple_shifted);
    let formant_ratio_simple = formant_energy_simple / total_energy_simple.max(1e-30);

    // The formant-preserving version should have at least as much relative
    // formant energy. We allow a small tolerance since the algorithm is
    // approximate.
    assert!(
        formant_ratio_fp >= formant_ratio_simple * 0.8,
        "formant preservation should maintain formant region energy ratio: \
         fp={formant_ratio_fp:.6}, simple={formant_ratio_simple:.6}"
    );
}

/// Estimate energy in a frequency band using a naive DFT.
fn band_energy(signal: &[f32], sr: f32, lo_hz: f32, hi_hz: f32) -> f64 {
    let n = signal.len();
    if n == 0 {
        return 0.0;
    }
    let n_bins = n / 2 + 1;
    let bin_hz = sr / n as f32;
    let lo_bin = (lo_hz / bin_hz).round() as usize;
    let hi_bin = (hi_hz / bin_hz).round().min(n_bins as f32 - 1.0) as usize;

    let two_pi_over_n = 2.0 * std::f64::consts::PI / n as f64;
    let mut energy = 0.0f64;

    for k in lo_bin..=hi_bin {
        let mut re: f64 = 0.0;
        let mut im: f64 = 0.0;
        for (i, &s) in signal.iter().enumerate() {
            if !s.is_finite() {
                continue;
            }
            let angle = two_pi_over_n * k as f64 * i as f64;
            re += f64::from(s) * angle.cos();
            im -= f64::from(s) * angle.sin();
        }
        energy += re * re + im * im;
    }
    energy
}

// ===========================================================================
// 8. Breath insertion
// ===========================================================================

#[test]
fn test_production_breath_insertion() {
    let n_voices = 3;
    let sr = KOKORO_SAMPLE_RATE as f32;

    // Create audio with a clear pause in the middle: 100ms signal, 100ms silence, 100ms signal.
    let signal_samples = (sr * 0.1) as usize;
    let pause_samples = (sr * 0.1) as usize;
    let total_samples = signal_samples * 2 + pause_samples;

    let mut template = Vec::with_capacity(total_samples);
    // First segment: sine.
    for i in 0..signal_samples {
        template.push(0.5 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr).sin());
    }
    // Pause: silence.
    template.extend(vec![0.0f32; pause_samples]);
    // Second segment: sine.
    for i in 0..signal_samples {
        template.push(0.5 * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / sr).sin());
    }

    let breath_config = BreathConfig::new()
        .with_noise_level(0.05)
        .with_duration_ms(50.0)
        .with_stagger_ms(10.0);
    breath_config.validate().expect("breath config valid");

    // Detect pauses in the template.
    let pauses = detect_pauses(&template, &breath_config);
    assert!(
        !pauses.is_empty(),
        "should detect at least one pause in the test signal"
    );

    // Create voices and insert breath.
    let mut voices: Vec<Vec<f32>> = (0..n_voices).map(|_| template.clone()).collect();
    let mut generator = BreathGenerator::new(&breath_config, n_voices).expect("breath generator");
    insert_breath_sounds(&mut voices, &pauses, &mut generator, &breath_config)
        .expect("insert breath");

    // The pause region should now have non-zero energy (breath noise was inserted).
    let pause_start = signal_samples;
    let pause_end = signal_samples + pause_samples;
    for (vi, voice) in voices.iter().enumerate() {
        let pause_slice = &voice[pause_start..pause_end.min(voice.len())];
        let pause_rms = rms(pause_slice);
        assert!(
            pause_rms > 1e-5,
            "voice {vi}: pause region should have breath noise, rms = {pause_rms}"
        );

        // All samples must remain finite.
        for (i, &s) in voice.iter().enumerate() {
            assert!(
                s.is_finite(),
                "voice {vi} sample {i} is not finite after breath insertion: {s}"
            );
        }
    }
}

// ===========================================================================
// 9. NaN safety
// ===========================================================================

#[test]
fn test_production_nan_safety() {
    let n = 2400;
    let n_voices = 3;

    // Create voices with various edge cases.
    let mut voices: Vec<Vec<f32>> = Vec::new();

    // Voice 0: normal signal.
    voices.push(make_sine(300.0, 0.1, 0.5));

    // Voice 1: denormals mixed with signal.
    let mut v1 = make_sine(350.0, 0.1, 0.5);
    // Inject denormals.
    for i in (0..v1.len()).step_by(100) {
        v1[i] = f32::MIN_POSITIVE / 2.0; // denormal
    }
    voices.push(v1);

    // Voice 2: large values that might stress limiter/dynamics.
    let v2: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / KOKORO_SAMPLE_RATE as f32;
            5.0 * (2.0 * std::f32::consts::PI * 400.0 * t).sin()
        })
        .collect();
    voices.push(v2);

    // Ensure all voices are the same length (pad shorter ones).
    let max_len = voices.iter().map(Vec::len).max().unwrap();
    for v in &mut voices {
        v.resize(max_len, 0.0);
    }

    let config = ChorusMasterConfig::full(n_voices).expect("full config");
    let (left, right) = process_chorus(&voices, &config).expect("NaN safety process");

    // Every output sample must be finite.
    for (i, &s) in left.iter().enumerate() {
        assert!(s.is_finite(), "NaN safety: left[{i}] = {s}");
    }
    for (i, &s) in right.iter().enumerate() {
        assert!(s.is_finite(), "NaN safety: right[{i}] = {s}");
    }
}

// ===========================================================================
// 10. Determinism
// ===========================================================================

#[test]
fn test_production_determinism() {
    let voices = make_voices(4, 220.0, 0.3);
    let config = ChorusMasterConfig::full(4).expect("full config");

    // Run the pipeline twice with fresh instances.
    let mut pipeline1 = ChorusMasterPipeline::new(config.clone()).expect("pipeline 1");
    let mut pipeline2 = ChorusMasterPipeline::new(config).expect("pipeline 2");

    let (l1, r1) = pipeline1.process(&voices).expect("run 1");
    let (l2, r2) = pipeline2.process(&voices).expect("run 2");

    assert_eq!(l1.len(), l2.len(), "output lengths must match");
    assert_eq!(r1.len(), r2.len(), "output lengths must match");

    let max_diff_l = l1
        .iter()
        .zip(l2.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_diff_r = r1
        .iter()
        .zip(r2.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff_l < 1e-6,
        "determinism violated: left max diff = {max_diff_l}"
    );
    assert!(
        max_diff_r < 1e-6,
        "determinism violated: right max diff = {max_diff_r}"
    );
}

// ===========================================================================
// Bonus: Pipeline reset isolation
// ===========================================================================

#[test]
fn test_production_pipeline_reset_isolation() {
    // Processing, resetting, then processing again should produce the same
    // result as two fresh pipelines. Use minimal config (no EQ/de-esser)
    // because the MixBusProcessor's biquad filters do not expose a reset
    // method, so configs with EQ/de-esser carry residual filter state.
    let voices = make_voices(3, 330.0, 0.2);
    let config = ChorusMasterConfig::minimal(3).expect("minimal config");

    let mut pipeline = ChorusMasterPipeline::new(config.clone()).expect("pipeline");
    let (l1, r1) = pipeline.process(&voices).expect("first run");

    pipeline.reset();
    let (l2, r2) = pipeline.process(&voices).expect("second run after reset");

    // After reset, the pipeline should behave identically to a fresh instance.
    let mut fresh_pipeline = ChorusMasterPipeline::new(config).expect("fresh pipeline");
    let (l_fresh, r_fresh) = fresh_pipeline.process(&voices).expect("fresh run");

    let max_diff_l = l2
        .iter()
        .zip(l_fresh.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_diff_r = r2
        .iter()
        .zip(r_fresh.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff_l < 1e-5,
        "reset isolation violated: left max diff = {max_diff_l}"
    );
    assert!(
        max_diff_r < 1e-5,
        "reset isolation violated: right max diff = {max_diff_r}"
    );

    // Also verify the first run is unchanged.
    for (i, &s) in l1.iter().enumerate() {
        assert!(s.is_finite(), "first run left[{i}] not finite: {s}");
    }
    for (i, &s) in r1.iter().enumerate() {
        assert!(s.is_finite(), "first run right[{i}] not finite: {s}");
    }
}

// ===========================================================================
// Diagnostic: Measure output levels for all presets (debug #4337)
// ===========================================================================

#[test]
fn test_diagnostic_output_levels_all_presets() {
    let n_voices = 4;
    let voices = make_voices(n_voices, 220.0, 0.5);
    let input_rms_val = rms(&voices[0]);
    let input_db = 20.0 * input_rms_val.log10();
    eprintln!("Input voice RMS: {input_rms_val:.4} ({input_db:.1} dB)");

    let preset_builders: Vec<(
        &str,
        Box<dyn Fn(usize) -> Result<ChorusMasterConfig, crate::kokoro_error::KokoroError>>,
    )> = vec![
        ("minimal", Box::new(ChorusMasterConfig::minimal)),
        ("standard", Box::new(ChorusMasterConfig::standard)),
        ("full", Box::new(ChorusMasterConfig::full)),
        (
            "singing_chorus",
            Box::new(ChorusMasterConfig::singing_chorus),
        ),
        (
            "speaking_chorus",
            Box::new(ChorusMasterConfig::speaking_chorus),
        ),
        ("intimate", Box::new(ChorusMasterConfig::intimate)),
        ("cathedral", Box::new(ChorusMasterConfig::cathedral)),
        ("broadcast", Box::new(ChorusMasterConfig::broadcast)),
    ];

    eprintln!("\n--- ChorusMasterConfig presets ---");
    let mut any_silent = false;
    for (name, builder) in &preset_builders {
        let config = builder(n_voices).unwrap();
        let (left, right) = process_chorus(&voices, &config).unwrap();
        let rms_l = rms(&left);
        let rms_r = rms(&right);
        let db_l = if rms_l > 0.0 {
            20.0 * rms_l.log10()
        } else {
            -999.0
        };
        let db_r = if rms_r > 0.0 {
            20.0 * rms_r.log10()
        } else {
            -999.0
        };
        eprintln!(
            "{name:20} L: {rms_l:.6} ({db_l:7.1} dB)  R: {rms_r:.6} ({db_r:7.1} dB)"
        );
        if db_l < -60.0 || db_r < -60.0 {
            any_silent = true;
            eprintln!("  *** SILENT OUTPUT DETECTED ***");
        }
    }

    eprintln!("\n--- ChorusPreset library presets ---");
    for &preset in ChorusPreset::ALL {
        let config = preset.to_config(n_voices).unwrap();
        let (left, right) = process_chorus(&voices, &config).unwrap();
        let rms_l = rms(&left);
        let rms_r = rms(&right);
        let db_l = if rms_l > 0.0 {
            20.0 * rms_l.log10()
        } else {
            -999.0
        };
        let db_r = if rms_r > 0.0 {
            20.0 * rms_r.log10()
        } else {
            -999.0
        };
        eprintln!(
            "{:20} L: {:.6} ({:7.1} dB)  R: {:.6} ({:7.1} dB)",
            preset.name(),
            rms_l,
            db_l,
            rms_r,
            db_r
        );
        if db_l < -60.0 || db_r < -60.0 {
            any_silent = true;
            eprintln!("  *** SILENT OUTPUT DETECTED ***");
        }
    }

    eprintln!("\n--- VocalChainPreset presets ---");
    for &preset in crate::kokoro_chorus_vocal_chain::VocalChainPreset::ALL {
        let config = preset.to_config(n_voices).unwrap();
        let (left, right) = process_chorus(&voices, &config).unwrap();
        let rms_l = rms(&left);
        let rms_r = rms(&right);
        let db_l = if rms_l > 0.0 {
            20.0 * rms_l.log10()
        } else {
            -999.0
        };
        let db_r = if rms_r > 0.0 {
            20.0 * rms_r.log10()
        } else {
            -999.0
        };
        eprintln!(
            "{:20} L: {:.6} ({:7.1} dB)  R: {:.6} ({:7.1} dB)",
            preset.name(),
            rms_l,
            db_l,
            rms_r,
            db_r
        );
        if db_l < -60.0 || db_r < -60.0 {
            any_silent = true;
            eprintln!("  *** SILENT OUTPUT DETECTED ***");
        }
    }

    assert!(
        !any_silent,
        "One or more presets produced silent output (< -60 dB)"
    );
}

// ===========================================================================
// Bonus: All ChorusMasterConfig presets at various voice counts
// ===========================================================================

#[test]
fn test_production_master_config_presets_all_voice_counts() {
    // Verify that all built-in ChorusMasterConfig presets (not ChorusPreset)
    // work for a range of voice counts.
    let preset_builders: Vec<(
        &str,
        Box<dyn Fn(usize) -> Result<ChorusMasterConfig, crate::kokoro_error::KokoroError>>,
    )> = vec![
        ("minimal", Box::new(ChorusMasterConfig::minimal)),
        ("standard", Box::new(ChorusMasterConfig::standard)),
        ("full", Box::new(ChorusMasterConfig::full)),
        (
            "singing_chorus",
            Box::new(ChorusMasterConfig::singing_chorus),
        ),
        (
            "speaking_chorus",
            Box::new(ChorusMasterConfig::speaking_chorus),
        ),
        ("intimate", Box::new(ChorusMasterConfig::intimate)),
        ("cathedral", Box::new(ChorusMasterConfig::cathedral)),
        ("broadcast", Box::new(ChorusMasterConfig::broadcast)),
    ];

    for &n in &[1, 2, 4, 8] {
        let voices = make_voices(n, 250.0, 0.1);
        for (name, builder) in &preset_builders {
            let config =
                builder(n).unwrap_or_else(|e| panic!("{name}({n}) construction failed: {e}"));
            config
                .validate()
                .unwrap_or_else(|e| panic!("{name}({n}) validation failed: {e}"));

            let (left, right) = process_chorus(&voices, &config)
                .unwrap_or_else(|e| panic!("{name}({n}) process failed: {e}"));

            for &s in &left {
                assert!(s.is_finite(), "{name}({n}): left has non-finite sample");
            }
            for &s in &right {
                assert!(s.is_finite(), "{name}({n}): right has non-finite sample");
            }
        }
    }
}

// ===========================================================================
// Diagnostic: dvoice-like Production preset with all advanced modules
// ===========================================================================

/// Construct a config that mirrors dvoice's `build_production` preset.
/// This enables ALL advanced modules (warmth, shimmer, depth staging, vocal
/// tract, harmonic tuner, adaptive dynamics, spectral matching, micro-pitch,
/// intelligibility, gain staging, decorrelation, vowel align, auto-mix, etc.)
/// to reproduce the -130 dB silent output bug (issue #4337).
fn build_dvoice_production_config(n_voices: usize) -> ChorusMasterConfig {
    use crate::kokoro_chorus_adaptive_dynamics::AdaptiveDynamicsConfig;
    use crate::kokoro_chorus_auto_mix::AutoMixConfig;
    use crate::kokoro_chorus_blend::EnsembleBlendConfig;
    use crate::kokoro_chorus_decorrelation::DecorrelationConfig;
    use crate::kokoro_chorus_depth_staging::DepthStagingConfig;
    use crate::kokoro_chorus_detune::DetuneConfig;
    use crate::kokoro_chorus_detune::DetuneDistribution;
    use crate::kokoro_chorus_dither::DitherConfig;
    use crate::kokoro_chorus_dynamics::DynamicsPreset;
    use crate::kokoro_chorus_eq::{DeEsserConfig, EqConfig};
    use crate::kokoro_chorus_exciter::ExciterConfig;
    use crate::kokoro_chorus_formant_tune::FormantTuneConfig;
    use crate::kokoro_chorus_gain_staging::GainStagingConfig;
    use crate::kokoro_chorus_harmonic_tuner::HarmonicTunerConfig;
    use crate::kokoro_chorus_humanize::HumanizeConfig;
    use crate::kokoro_chorus_intelligibility::IntelligibilityConfig;
    use crate::kokoro_chorus_intonation::IntonationConfig;
    use crate::kokoro_chorus_micro_pitch::MicroPitchConfig;
    use crate::kokoro_chorus_onset_sync::OnsetSyncConfig;
    use crate::kokoro_chorus_oversample::OversampleConfig;
    use crate::kokoro_chorus_reverb::ReverbConfig;
    use crate::kokoro_chorus_shimmer::ShimmerConfig;
    use crate::kokoro_chorus_spectral_match::SpectralMatchConfig;
    use crate::kokoro_chorus_stereo::StereoChorusConfig;
    use crate::kokoro_chorus_vibrato::VibratoConfig;
    use crate::kokoro_chorus_vocal_tract::VocalTractConfig;
    use crate::kokoro_chorus_voice_alloc::VoiceAllocConfig;
    use crate::kokoro_chorus_vowel_align::VowelAlignConfig;
    use crate::kokoro_chorus_warmth::{WarmthConfig, WarmthMode};
    use crate::kokoro_chorus_width::StereoWidthConfig;

    let mut config = ChorusMasterConfig::new(n_voices).unwrap();

    // Per-voice EQ
    config.eq = Some(EqConfig {
        low_freq: 180.0,
        low_gain_db: 1.0,
        mid_freq: 2800.0,
        mid_gain_db: 1.5,
        mid_q: 0.9,
        high_freq: 9000.0,
        high_gain_db: -1.0,
    });

    // De-essing
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 6000.0,
        q: 1.2,
        threshold_db: -20.0,
        max_reduction_db: -10.0,
        attack_sec: 0.001,
        release_sec: 0.050,
    });

    // Vibrato
    config.vibrato = Some(VibratoConfig {
        rate_hz: 5.5,
        depth_cents: 25.0,
        rate_spread_hz: 0.4,
        depth_spread_cents: 8.0,
        onset_sec: 0.18,
    });

    // Detuning
    config.detune = Some(DetuneConfig {
        cents_spread: 10.0,
        distribution: DetuneDistribution::Gaussian,
        seed: 0,
    });

    // Micro-pitch drift
    config.micro_pitch = Some(MicroPitchConfig::default());

    // Humanization
    config.humanize = Some(HumanizeConfig::default());

    // Blending
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.45,
        formant_preservation: true,
        harmonic_alignment: true,
        min_period: 30,
        max_period: 300,
    });

    // Stereo
    config.stereo = Some(
        StereoChorusConfig::auto_layout(n_voices)
            .unwrap()
            .with_stereo_width(0.75),
    );

    // Width
    config.width = Some(
        StereoWidthConfig::new()
            .with_width(1.15)
            .with_bass_mono_freq(80.0),
    );

    // Warmth
    config.warmth = Some(
        WarmthConfig::new()
            .with_warmth_amount(0.25)
            .with_presence_amount(0.2)
            .with_warmth_mode(WarmthMode::Tube),
    );

    // Shimmer
    config.shimmer = Some(
        ShimmerConfig::new()
            .with_shimmer_amount(0.2)
            .with_air_gain_db(2.0)
            .with_brightness(0.4),
    );

    // Depth staging
    config.depth_staging = Some(
        DepthStagingConfig::new()
            .with_n_voices(n_voices)
            .with_depth_spread(0.5)
            .with_lead_voice_depth(0.1)
            .with_backing_voice_depth(0.5),
    );

    // Vocal tract
    config.vocal_tract = Some(VocalTractConfig::new(n_voices).unwrap());

    // Harmonic tuner
    config.harmonic_tuner = Some(HarmonicTunerConfig::default());

    // Spectral match
    config.spectral_match = Some(SpectralMatchConfig::default());

    // Intelligibility
    config.intelligibility = Some(IntelligibilityConfig::default());

    // Exciter
    config.exciter = Some(
        ExciterConfig::new()
            .with_harmonics_mix(0.15)
            .with_air_gain_db(1.0),
    );

    // Decorrelation
    config.decorrelation = Some(DecorrelationConfig::default());

    // Vowel align
    config.vowel_align = Some(VowelAlignConfig::default());

    // Formant tune
    config.formant_tune = Some(FormantTuneConfig::default());

    // Intonation
    config.intonation = Some(IntonationConfig::default());

    // Onset sync
    config.onset_sync = Some(OnsetSyncConfig::default());

    // Voice allocation
    config.voice_alloc = Some(VoiceAllocConfig::default());

    // Auto-mix
    config.auto_mix = Some(AutoMixConfig::default());

    // Oversampling
    config.oversample = Some(OversampleConfig::default());

    // Adaptive dynamics
    config.adaptive_dynamics = Some(AdaptiveDynamicsConfig::default());

    // Regular dynamics
    config.dynamics = Some(DynamicsPreset::Mastering.to_config());

    // Reverb
    config.reverb = Some(ReverbConfig {
        reverb_mix: 0.15,
        room_size: 0.40,
        early_reflections: true,
        damping: 0.50,
    });

    // Gain staging
    config.gain_staging = Some(GainStagingConfig::default());

    // Dither
    config.dither = Some(DitherConfig::default());

    // Limiter
    config.limiter_enabled = true;

    config
}

#[test]
fn test_dvoice_production_preset_output_level() {
    let n_voices = 4;
    let voices = make_voices(n_voices, 220.0, 0.5);
    let input_rms = rms(&voices[0]);
    let input_db = 20.0 * input_rms.log10();

    let config = build_dvoice_production_config(n_voices);
    let mut pipeline =
        ChorusMasterPipeline::new(config).expect("dvoice production config should be valid");
    let (left, right) = pipeline
        .process(&voices)
        .expect("pipeline processing should succeed");

    let left_rms = rms(&left);
    let right_rms = rms(&right);
    let out_rms = f32::midpoint(left_rms, right_rms);
    let out_db = if out_rms > 1e-12 {
        20.0 * out_rms.log10()
    } else {
        -120.0
    };

    eprintln!("=== dvoice Production preset diagnostic ===");
    eprintln!("Input  RMS: {input_rms:.6} ({input_db:.1} dB)");
    eprintln!(
        "Left   RMS: {left_rms:.6} ({:.1} dB)",
        20.0 * left_rms.max(1e-12).log10()
    );
    eprintln!(
        "Right  RMS: {right_rms:.6} ({:.1} dB)",
        20.0 * right_rms.max(1e-12).log10()
    );
    eprintln!("Output RMS: {out_rms:.6} ({out_db:.1} dB)");
    eprintln!("Gain: {:.1} dB", out_db - input_db);

    // Check for the -130 dB bug: output should be within 40 dB of input
    assert!(
        out_db > input_db - 40.0,
        "OUTPUT TOO QUIET: {out_db:.1} dB (input was {input_db:.1} dB, \
         loss = {:.1} dB). This reproduces issue #4337.",
        input_db - out_db,
    );
}

/// Bisection test: start with singing_chorus, progressively add Production
/// modules to find which one causes the level drop.
#[test]
fn test_bisect_production_modules_for_level_drop() {
    let n_voices = 4;
    let voices = make_voices(n_voices, 220.0, 0.5);
    let input_rms = rms(&voices[0]);
    let input_db = 20.0 * input_rms.log10();

    eprintln!("=== Module bisection: finding level drop ===");
    eprintln!("Input RMS: {input_rms:.6} ({input_db:.1} dB)\n");

    // Baseline: singing_chorus
    let config = ChorusMasterConfig::singing_chorus(n_voices).unwrap();
    let mut pipeline = ChorusMasterPipeline::new(config).unwrap();
    let (l, r) = pipeline.process(&voices).unwrap();
    let base_db = 20.0 * f32::midpoint(rms(&l), rms(&r)).max(1e-12).log10();
    eprintln!("singing_chorus baseline: {base_db:.1} dB");

    // Now add modules one at a time
    let module_names = [
        "micro_pitch",
        "warmth",
        "shimmer",
        "depth_staging",
        "vocal_tract",
        "harmonic_tuner",
        "spectral_match",
        "intelligibility",
        "exciter",
        "decorrelation",
        "vowel_align",
        "formant_tune",
        "intonation",
        "onset_sync",
        "voice_alloc",
        "auto_mix",
        "oversample",
        "adaptive_dynamics",
        "gain_staging",
        "dither",
        "width",
    ];

    for name in &module_names {
        let mut config = ChorusMasterConfig::singing_chorus(n_voices).unwrap();

        // Add the specific module
        match *name {
            "micro_pitch" => {
                config.micro_pitch =
                    Some(crate::kokoro_chorus_micro_pitch::MicroPitchConfig::default());
            }
            "warmth" => {
                config.warmth = Some(
                    crate::kokoro_chorus_warmth::WarmthConfig::new()
                        .with_warmth_amount(0.25)
                        .with_warmth_mode(crate::kokoro_chorus_warmth::WarmthMode::Tube),
                );
            }
            "shimmer" => {
                config.shimmer = Some(crate::kokoro_chorus_shimmer::ShimmerConfig::new());
            }
            "depth_staging" => {
                config.depth_staging = Some(
                    crate::kokoro_chorus_depth_staging::DepthStagingConfig::new()
                        .with_n_voices(n_voices)
                        .with_depth_spread(0.5),
                );
            }
            "vocal_tract" => {
                config.vocal_tract = Some(
                    crate::kokoro_chorus_vocal_tract::VocalTractConfig::new(n_voices).unwrap(),
                );
            }
            "harmonic_tuner" => {
                config.harmonic_tuner =
                    Some(crate::kokoro_chorus_harmonic_tuner::HarmonicTunerConfig::default());
            }
            "spectral_match" => {
                config.spectral_match =
                    Some(crate::kokoro_chorus_spectral_match::SpectralMatchConfig::default());
            }
            "intelligibility" => {
                config.intelligibility =
                    Some(crate::kokoro_chorus_intelligibility::IntelligibilityConfig::default());
            }
            "exciter" => {
                config.exciter = Some(crate::kokoro_chorus_exciter::ExciterConfig::new());
            }
            "decorrelation" => {
                config.decorrelation =
                    Some(crate::kokoro_chorus_decorrelation::DecorrelationConfig::default());
            }
            "vowel_align" => {
                config.vowel_align =
                    Some(crate::kokoro_chorus_vowel_align::VowelAlignConfig::default());
            }
            "formant_tune" => {
                config.formant_tune =
                    Some(crate::kokoro_chorus_formant_tune::FormantTuneConfig::default());
            }
            "intonation" => {
                config.intonation =
                    Some(crate::kokoro_chorus_intonation::IntonationConfig::default());
            }
            "onset_sync" => {
                config.onset_sync =
                    Some(crate::kokoro_chorus_onset_sync::OnsetSyncConfig::default());
            }
            "voice_alloc" => {
                config.voice_alloc =
                    Some(crate::kokoro_chorus_voice_alloc::VoiceAllocConfig::default());
            }
            "auto_mix" => {
                config.auto_mix = Some(crate::kokoro_chorus_auto_mix::AutoMixConfig::default());
            }
            "oversample" => {
                config.oversample =
                    Some(crate::kokoro_chorus_oversample::OversampleConfig::default());
            }
            "adaptive_dynamics" => {
                config.adaptive_dynamics =
                    Some(crate::kokoro_chorus_adaptive_dynamics::AdaptiveDynamicsConfig::default());
            }
            "gain_staging" => {
                config.gain_staging =
                    Some(crate::kokoro_chorus_gain_staging::GainStagingConfig::default());
            }
            "dither" => {
                config.dither = Some(crate::kokoro_chorus_dither::DitherConfig::default());
            }
            "width" => {
                config.width =
                    Some(crate::kokoro_chorus_width::StereoWidthConfig::new().with_width(1.15));
            }
            _ => {}
        }

        let mut pipeline = ChorusMasterPipeline::new(config).unwrap();
        let (l, r) = pipeline.process(&voices).unwrap();
        let out_db = 20.0 * f32::midpoint(rms(&l), rms(&r)).max(1e-12).log10();
        let delta = out_db - base_db;
        let flag = if delta < -10.0 { " *** SUSPICIOUS" } else { "" };
        eprintln!("  +{name}: {out_db:.1} dB (delta: {delta:+.1} dB){flag}");
    }
}
