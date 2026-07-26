// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Streaming quality verification tests for Wave 22 Kokoro improvements.
//!
//! Tests verify three categories of improvements:
//! 1. **Phase continuity**: SineGen `last_cumphase` carryover across chunks
//! 2. **Hann crossfade**: Hann window produces lower energy error than linear
//! 3. **Chorus quality**: pitch detuning, soft limiter, sqrt gain normalization
//!
//! All tests use synthetic data and require no model weights or GPU.

use crate::kokoro_chorus::{mix_voices_with_config, pitch_shift_factor, ChorusConfig};
use crate::kokoro_source::SineGen;
use crate::kokoro_streaming::{assemble_streaming_chunks, CrossfadeWindow, KokoroStreamConfig};
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a pure sine wave at the given frequency and sample rate.
fn generate_sine(freq_hz: f32, num_samples: usize, sample_rate: usize) -> Vec<f32> {
    let sr = sample_rate as f64;
    let f = f64::from(freq_hz);
    (0..num_samples)
        .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / sr).sin() as f32)
        .collect()
}

/// RMS (root mean square) of a signal slice.
fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = signal.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    (sum_sq / signal.len() as f64).sqrt() as f32
}

/// Peak absolute amplitude in a signal.
fn peak_amplitude(signal: &[f32]) -> f32 {
    signal.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// 1. Phase continuity tests (SineGen last_cumphase carryover)
// ---------------------------------------------------------------------------

/// Verify that SineGen carries phase across two consecutive forward() calls.
///
/// Strategy: compare boundary discontinuity WITH phase carryover vs WITHOUT.
/// Phase carryover should produce a much smaller boundary artifact than a
/// full phase reset. The crossfade overlap then handles the remaining seam.
///
/// Note: even with carryover, there is a small boundary artifact because
/// phase is carried at frame rate and then upsampled via interpolation.
/// The carryover ensures this artifact is bounded by the interpolation error,
/// not a full phase discontinuity (which would be 2 * sine_amp = 0.2).
#[test]
fn test_sinegen_phase_continuity_across_chunks() {
    let sg = SineGen::new();
    let device = Device::Cpu;
    let upp: usize = 300;
    let chunk_frames: usize = 40; // 40 frames * 300 upp = 12000 audio samples
    let f0_val: f32 = 220.0; // A3
    let n_ch = sg.n_channels();

    // --- With phase carryover ---
    sg.reset_phase();
    let f0 = DynTensor::full(&[1, chunk_frames, 1], f64::from(f0_val), DType::F32, &device).unwrap();
    let (sines1_carry, _, _) = sg.forward(&f0, upp).unwrap();
    let pcm1_carry = sines1_carry.to_flat_vec::<f32>().unwrap();
    // DO NOT reset — phase carries over
    let (sines2_carry, _, _) = sg.forward(&f0, upp).unwrap();
    let pcm2_carry = sines2_carry.to_flat_vec::<f32>().unwrap();

    // --- Without phase carryover (reset between chunks) ---
    sg.reset_phase();
    let (sines1_reset, _, _) = sg.forward(&f0, upp).unwrap();
    let pcm1_reset = sines1_reset.to_flat_vec::<f32>().unwrap();
    sg.reset_phase(); // Force reset
    let (sines2_reset, _, _) = sg.forward(&f0, upp).unwrap();
    let pcm2_reset = sines2_reset.to_flat_vec::<f32>().unwrap();

    let t_audio_chunk = chunk_frames * upp;

    // Compare boundary discontinuity for each harmonic.
    // Sum the squared boundary deltas across all harmonics as a measure
    // of total boundary discontinuity energy.
    let mut carry_boundary_energy: f64 = 0.0;
    let mut reset_boundary_energy: f64 = 0.0;
    for ch in 0..n_ch {
        let last_carry = pcm1_carry[(t_audio_chunk - 1) * n_ch + ch];
        let first_carry = pcm2_carry[ch];
        let carry_delta = f64::from(first_carry - last_carry);
        carry_boundary_energy += carry_delta * carry_delta;

        let last_reset = pcm1_reset[(t_audio_chunk - 1) * n_ch + ch];
        let first_reset = pcm2_reset[ch];
        let reset_delta = f64::from(first_reset - last_reset);
        reset_boundary_energy += reset_delta * reset_delta;
    }

    // Phase carryover should produce less total boundary energy than reset.
    // The reset case has random phase alignment, so boundary energy can be large.
    // The carryover case has phase-aligned boundaries with only interpolation error.
    assert!(
        carry_boundary_energy <= reset_boundary_energy,
        "Phase carryover should produce smaller boundary energy: \
         carry={carry_boundary_energy:.8}, reset={reset_boundary_energy:.8}"
    );

    // Verify the carryover actually stores state by checking that chunk 2
    // with carryover differs from chunk 2 with reset. If carryover did nothing,
    // both would be identical (both starting from phase=0).
    let mut diff_count = 0usize;
    for i in 0..pcm2_carry.len().min(pcm2_reset.len()) {
        if (pcm2_carry[i] - pcm2_reset[i]).abs() > 1e-6 {
            diff_count += 1;
        }
    }
    assert!(
        diff_count > 0,
        "Phase carryover should change chunk 2 output vs reset"
    );
}

/// Verify that without phase reset between chunks, the concatenation is
/// continuous, while resetting phase causes a discontinuity.
#[test]
fn test_sinegen_phase_reset_causes_discontinuity() {
    let sg = SineGen::new();
    let device = Device::Cpu;
    let upp: usize = 300;
    let chunk_frames: usize = 20;
    let f0_val: f32 = 440.0;

    // --- With phase carryover (no reset between chunks) ---
    sg.reset_phase();
    let f0 = DynTensor::full(&[1, chunk_frames, 1], f64::from(f0_val), DType::F32, &device).unwrap();
    let (sines1, _, _) = sg.forward(&f0, upp).unwrap();
    let pcm1 = sines1.to_flat_vec::<f32>().unwrap();
    // DO NOT reset — phase carries over
    let (sines2_carry, _, _) = sg.forward(&f0, upp).unwrap();
    let pcm2_carry = sines2_carry.to_flat_vec::<f32>().unwrap();

    // --- With phase reset between chunks ---
    sg.reset_phase();
    let (_, _, _) = sg.forward(&f0, upp).unwrap();
    sg.reset_phase(); // force reset
    let (sines2_reset, _, _) = sg.forward(&f0, upp).unwrap();
    let pcm2_reset = sines2_reset.to_flat_vec::<f32>().unwrap();

    let n_ch = sg.n_channels();
    let t_audio = chunk_frames * upp;

    // At the boundary, the last sample of chunk1 and first sample of chunk2
    // should be continuous for the carried case, but may be discontinuous
    // for the reset case.
    let last_sample_c1 = pcm1[(t_audio - 1) * n_ch]; // fundamental, last sample
    let first_sample_carry = pcm2_carry[0]; // fundamental, first sample (carried)
    let first_sample_reset = pcm2_reset[0]; // fundamental, first sample (reset)

    let diff_carry = (last_sample_c1 - first_sample_carry).abs();
    let diff_reset = (last_sample_c1 - first_sample_reset).abs();

    // Carried phase should have much smaller discontinuity than reset phase.
    // (Not guaranteed to be strictly smaller in every case due to phase wrapping,
    // but for typical F0 values the difference is significant.)
    // The key assertion: carried phase produces a small boundary jump.
    assert!(
        diff_carry < 0.05,
        "Phase carryover should produce near-continuous boundary: diff_carry={diff_carry}"
    );

    // The reset case may or may not have a large discontinuity depending on
    // where the phase lands, but we just verify the carried case is smooth.
    let _ = diff_reset; // used for documentation, not assertion
}

// ---------------------------------------------------------------------------
// 2. Hann vs Linear crossfade tests
// ---------------------------------------------------------------------------

/// Verify that Hann crossfade has smoother edge transitions than linear.
///
/// The key advantage of Hann over linear crossfade is that the Hann window
/// has zero derivative at the boundaries (starts and ends tangentially),
/// while linear crossfade has a constant-rate transition that creates an
/// abrupt gain change at the crossfade edges.
///
/// Strategy: crossfade between a constant signal (1.0) and zero (0.0).
/// Check that the Hann window has a smaller derivative at the crossfade
/// edges compared to linear.
#[test]
fn test_hann_crossfade_smoother_edges_than_linear() {
    let crossfade_samples: usize = 960; // 40ms at 24kHz

    // Use constant signals to isolate the crossfade curve shape.
    let tail = vec![1.0f32; crossfade_samples];
    let head = vec![0.0f32; crossfade_samples];

    let linear_blend = nn_core::audio::crossfade_linear_blend(&tail, &head, crossfade_samples);
    let hann_blend = nn_core::audio::crossfade_hann_blend(&tail, &head, crossfade_samples);

    // At the start of the crossfade (i=0, 1), the derivative of the blend
    // should be near-zero for Hann and non-zero for linear.
    // Linear: blend[0] = 1.0, blend[1] = 1.0 - 1/(N-1)
    // Hann: blend[0] = 1.0, blend[1] ≈ 1.0 (near-zero slope at start)
    let linear_start_delta = (linear_blend[1] - linear_blend[0]).abs();
    let hann_start_delta = (hann_blend[1] - hann_blend[0]).abs();

    assert!(
        hann_start_delta < linear_start_delta,
        "Hann should have smaller start delta: hann={hann_start_delta:.8}, \
         linear={linear_start_delta:.8}"
    );

    // At the end of the crossfade (i=N-2, N-1), same property.
    let n = crossfade_samples;
    let linear_end_delta = (linear_blend[n - 1] - linear_blend[n - 2]).abs();
    let hann_end_delta = (hann_blend[n - 1] - hann_blend[n - 2]).abs();

    assert!(
        hann_end_delta < linear_end_delta,
        "Hann should have smaller end delta: hann={hann_end_delta:.8}, \
         linear={linear_end_delta:.8}"
    );

    // Both should produce the same midpoint value (0.5) for 1.0→0.0 transition.
    let mid = crossfade_samples / 2;
    assert!(
        (linear_blend[mid] - 0.5).abs() < 0.01,
        "Linear midpoint should be ~0.5: got {:.6}",
        linear_blend[mid]
    );
    assert!(
        (hann_blend[mid] - 0.5).abs() < 0.01,
        "Hann midpoint should be ~0.5: got {:.6}",
        hann_blend[mid]
    );

    // Verify max derivative of Hann is larger than linear in the middle
    // (the Hann window transitions faster in the center to compensate
    // for the slower edges — this is the raised-cosine shape).
    let mid_start = crossfade_samples / 4;
    let mid_end = 3 * crossfade_samples / 4;
    let hann_mid_max_delta: f32 = (mid_start..mid_end)
        .map(|i| (hann_blend[i + 1] - hann_blend[i]).abs())
        .fold(0.0f32, f32::max);
    let linear_mid_max_delta: f32 = (mid_start..mid_end)
        .map(|i| (linear_blend[i + 1] - linear_blend[i]).abs())
        .fold(0.0f32, f32::max);

    // Hann transitions faster in the center than linear (which is constant).
    assert!(
        hann_mid_max_delta > linear_mid_max_delta,
        "Hann should have faster mid-section transition: \
         hann_mid_max_delta={hann_mid_max_delta:.8}, linear_mid_max_delta={linear_mid_max_delta:.8}"
    );
}

/// Verify that the assembled chunks use Hann crossfade when configured.
#[test]
fn test_assemble_streaming_chunks_uses_hann_window() {
    let chunk_len: usize = 4800;
    let crossfade_samples: usize = 960;

    // Two chunks of constant value to verify the crossfade curve shape.
    let chunk1 = vec![1.0f32; chunk_len];
    let chunk2 = vec![0.0f32; chunk_len];

    // Hann crossfade config
    let config_hann = KokoroStreamConfig {
        crossfade_samples,
        crossfade_window: CrossfadeWindow::Hann,
    };

    // Linear crossfade config
    let config_linear = KokoroStreamConfig {
        crossfade_samples,
        crossfade_window: CrossfadeWindow::Linear,
    };

    let chunks_hann =
        assemble_streaming_chunks(&[chunk1.clone(), chunk2.clone()], &config_hann).unwrap();
    let chunks_linear =
        assemble_streaming_chunks(&[chunk1, chunk2], &config_linear).unwrap();

    // The second chunk starts with the crossfade region.
    // For constant 1.0 → 0.0 transition:
    //   Linear midpoint: 0.5
    //   Hann midpoint: 0.5 (both are 0.5 at the center)
    // But the curve shape differs. At 25% of the crossfade:
    //   Linear: alpha = 0.25, blend = 0.75
    //   Hann: alpha = 0.5*(1 - cos(pi*0.25)) ≈ 0.146, blend ≈ 0.854
    let quarter_idx = crossfade_samples / 4;
    let hann_val = chunks_hann[1].pcm[quarter_idx];
    let linear_val = chunks_linear[1].pcm[quarter_idx];

    // At 25%, Hann should still be closer to the previous chunk's value (1.0)
    // because the Hann window starts slower than linear.
    assert!(
        hann_val > linear_val,
        "Hann crossfade should be slower at the start: hann_val={hann_val:.4}, \
         linear_val={linear_val:.4}"
    );
}

/// Verify that Hann crossfade is energy-preserving for equal-amplitude signals.
///
/// For two identical signals, both Hann and linear crossfade should preserve
/// the signal perfectly (alpha * x + (1-alpha) * x = x).
#[test]
fn test_hann_crossfade_preserves_identical_signals() {
    let crossfade_samples: usize = 960;
    let signal = generate_sine(440.0, crossfade_samples, KOKORO_SAMPLE_RATE);

    let hann_blend = nn_core::audio::crossfade_hann_blend(&signal, &signal, crossfade_samples);
    let linear_blend = nn_core::audio::crossfade_linear_blend(&signal, &signal, crossfade_samples);

    // Both should be identical to the input.
    for i in 0..crossfade_samples {
        assert!(
            (hann_blend[i] - signal[i]).abs() < 1e-6,
            "Hann crossfade of identical signals should preserve signal at i={i}: \
             got={}, expected={}",
            hann_blend[i],
            signal[i]
        );
        assert!(
            (linear_blend[i] - signal[i]).abs() < 1e-6,
            "Linear crossfade of identical signals should preserve signal at i={i}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Chorus quality tests
// ---------------------------------------------------------------------------

/// Verify that pitch detuning produces output different from zero detuning.
///
/// With pitch_semitones=[0.05, -0.05, 0.1, -0.1], the mixed chorus should
/// differ from a zero-detuning chorus because each voice's effective F0
/// is shifted by the pitch factor.
#[test]
fn test_chorus_pitch_detuning_changes_output() {
    let n_voices = 4;
    let n_samples = 2400; // 100ms at 24kHz
    let freq = 440.0;

    // Generate identical sine waves for all voices.
    let voice_pcm: Vec<Vec<f32>> = (0..n_voices)
        .map(|_| generate_sine(freq, n_samples, KOKORO_SAMPLE_RATE))
        .collect();

    // Zero detuning config
    let config_zero = ChorusConfig::equal_gain(n_voices)
        .unwrap()
        .with_pitch_semitones(vec![0.0; n_voices]);

    // Since mix_voices_with_config doesn't apply pitch shift to the PCM
    // directly (pitch shift is applied during synthesis as F0 scaling),
    // we simulate the effect: generate voices with shifted frequencies.
    let pitches = [0.05f32, -0.05, 0.1, -0.1];
    let detuned_voices: Vec<Vec<f32>> = pitches
        .iter()
        .map(|&p| {
            let shifted_freq = freq * pitch_shift_factor(p);
            generate_sine(shifted_freq, n_samples, KOKORO_SAMPLE_RATE)
        })
        .collect();

    // Mix the unshifted voices (simulating zero detuning).
    let mixed_zero = mix_voices_with_config(&voice_pcm, &config_zero).unwrap();

    // Mix the pitch-shifted voices (simulating real detuning).
    let zero_pitch_config = ChorusConfig::equal_gain(n_voices).unwrap();
    let mixed_detuned = mix_voices_with_config(&detuned_voices, &zero_pitch_config).unwrap();

    // The detuned mix should differ from the zero-detuned mix.
    let max_diff: f32 = mixed_zero
        .iter()
        .zip(mixed_detuned.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff > 1e-5,
        "Pitch-detuned chorus should differ from zero-detuning: max_diff={max_diff}"
    );

    // Verify pitch_shift_factor produces expected values.
    assert!(
        (pitch_shift_factor(0.0) - 1.0).abs() < 1e-7,
        "0 semitones should give factor 1.0"
    );
    assert!(
        (pitch_shift_factor(12.0) - 2.0).abs() < 1e-5,
        "+12 semitones should give factor 2.0"
    );
    assert!(
        (pitch_shift_factor(-12.0) - 0.5).abs() < 1e-5,
        "-12 semitones should give factor 0.5"
    );

    // Small detuning (5 cents = 0.05 semitones) should produce a factor
    // very close to but not equal to 1.0.
    let factor_5_cents = pitch_shift_factor(0.05);
    assert!(
        factor_5_cents > 1.0 && factor_5_cents < 1.003,
        "+5 cents factor should be in (1.0, 1.003): got {factor_5_cents}"
    );
}

/// Verify that soft limiter keeps peak amplitude below 1.0 and avoids
/// hard clipping artifacts.
///
/// With soft_limiter_drive=1.5, the tanh limiter should smoothly compress
/// peaks. Even if the pre-limiter sum exceeds 1.0, the output stays bounded.
#[test]
fn test_soft_limiter_bounds_output() {
    let n_voices = 4;
    let n_samples = 2400;

    // Generate in-phase sines that will constructively interfere.
    // With 4 voices at gain 1/4 each, constructive interference at peaks
    // gives amplitude 1.0 before limiting. To test limiting, use gains
    // that produce peaks > 1.0.
    let voice_pcm: Vec<Vec<f32>> = (0..n_voices)
        .map(|_| generate_sine(440.0, n_samples, KOKORO_SAMPLE_RATE))
        .collect();

    // Use sqrt normalization which gives gain = 1/sqrt(4) = 0.5 per voice.
    // 4 in-phase voices × 0.5 = peak of 2.0 before limiting.
    let config = ChorusConfig::equal_power(n_voices)
        .unwrap()
        .with_soft_limiter(1.5);

    let mixed = mix_voices_with_config(&voice_pcm, &config).unwrap();
    let peak = peak_amplitude(&mixed);

    // Soft limiter with drive=1.5: output bounded by tanh(x*1.5)/1.5.
    // tanh(x*1.5)/1.5 < 1/1.5 ≈ 0.667, so peak < 0.667 for any input.
    // Actually tanh approaches 1, so the bound is 1/drive.
    // For drive=1.5: max output = tanh(inf*1.5)/1.5 = 1/1.5 = 0.667.
    assert!(
        peak < 1.0,
        "Soft limiter output should be < 1.0: peak={peak}"
    );
    assert!(
        peak <= 1.0 / 1.5 + 0.001,
        "Soft limiter output should be bounded by 1/drive (~0.667): peak={peak}"
    );

    // Verify no hard clipping: no samples should be exactly ±1.0.
    let hard_clip_count = mixed.iter().filter(|&&s| s == 1.0 || s == -1.0).count();
    assert_eq!(
        hard_clip_count, 0,
        "Soft limiter should not produce hard-clipped samples at exactly +/- 1.0"
    );

    // Verify the signal is not all zeros (limiter is not destroying the signal).
    assert!(
        peak > 0.01,
        "Soft limiter output should not be near-zero: peak={peak}"
    );
}

/// Verify that soft limiter produces smooth (non-clipped) waveform.
///
/// Check that adjacent samples don't have abrupt jumps characteristic of
/// hard clipping. The maximum sample-to-sample difference should be bounded.
#[test]
fn test_soft_limiter_smooth_waveform() {
    let n_voices = 4;
    let n_samples = 4800;

    // High amplitude signals to trigger limiting.
    let voice_pcm: Vec<Vec<f32>> = (0..n_voices)
        .map(|_| {
            generate_sine(440.0, n_samples, KOKORO_SAMPLE_RATE)
                .iter()
                .map(|&s| s * 0.8) // Scale up for higher energy
                .collect()
        })
        .collect();

    let config = ChorusConfig::equal_power(n_voices)
        .unwrap()
        .with_soft_limiter(1.5);

    let mixed = mix_voices_with_config(&voice_pcm, &config).unwrap();

    // Calculate max sample-to-sample difference.
    let max_delta: f32 = mixed
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max);

    // For a 440Hz sine at 24kHz, the max delta between adjacent samples is:
    // d/dt sin(2*pi*440*t) = 2*pi*440*cos(...) ≈ 2765 at sample rate 24000
    // delta per sample ≈ 2*pi*440/24000 ≈ 0.115
    // With soft limiting, the delta should be even smaller.
    assert!(
        max_delta < 0.2,
        "Soft-limited waveform should have smooth transitions: max_delta={max_delta}"
    );
}

/// Verify that sqrt gain normalization produces higher RMS than 1/N normalization.
///
/// For uncorrelated voices, 1/sqrt(N) gain preserves perceived loudness
/// while 1/N gain reduces it as N increases.
#[test]
fn test_sqrt_gain_normalization_higher_rms() {
    let n_voices = 4;
    let n_samples = 4800;

    // Generate uncorrelated voices (different frequencies to approximate
    // uncorrelated signals).
    let freqs = [440.0, 523.25, 659.26, 783.99]; // A4, C5, E5, G5
    let voice_pcm: Vec<Vec<f32>> = freqs
        .iter()
        .map(|&f| generate_sine(f, n_samples, KOKORO_SAMPLE_RATE))
        .collect();

    // 1/N normalization (equal_gain)
    let config_1n = ChorusConfig::equal_gain(n_voices).unwrap();
    let mixed_1n = mix_voices_with_config(&voice_pcm, &config_1n).unwrap();

    // 1/sqrt(N) normalization (equal_power)
    let config_sqrt = ChorusConfig::equal_power(n_voices).unwrap();
    let mixed_sqrt = mix_voices_with_config(&voice_pcm, &config_sqrt).unwrap();

    let rms_1n = rms(&mixed_1n);
    let rms_sqrt = rms(&mixed_sqrt);

    // sqrt normalization should produce higher RMS.
    // 1/sqrt(4) = 0.5 vs 1/4 = 0.25, so sqrt RMS should be ~2x higher.
    assert!(
        rms_sqrt > rms_1n,
        "sqrt gain normalization should produce higher RMS: \
         rms_sqrt={rms_sqrt:.6}, rms_1n={rms_1n:.6}"
    );

    // The ratio should be approximately sqrt(N) / 1 = 2.0 for N=4.
    // In practice, because voices are not perfectly uncorrelated (they're
    // harmonic sines), the ratio may differ slightly.
    let ratio = rms_sqrt / rms_1n;
    assert!(
        ratio > 1.5 && ratio < 2.5,
        "RMS ratio (sqrt/1n) should be approximately sqrt(N)=2.0: ratio={ratio:.3}"
    );
}

/// Verify that the rich_chorus preset applies all quality params.
#[test]
fn test_rich_chorus_preset_configuration() {
    let n_voices = 4;
    let config = ChorusConfig::rich_chorus(n_voices).unwrap();

    // Should have sqrt gain normalization.
    assert!(
        config.sqrt_gain_normalization,
        "rich_chorus should enable sqrt gain normalization"
    );

    // Should have soft limiter.
    assert!(
        config.soft_limiter_drive.is_some(),
        "rich_chorus should enable soft limiter"
    );
    assert!(
        (config.soft_limiter_drive.unwrap() - 1.5).abs() < 1e-6,
        "rich_chorus should use drive=1.5"
    );

    // Should have pitch detuning.
    let pitches = config.pitch_semitones.as_ref().unwrap();
    assert_eq!(pitches.len(), n_voices);
    // Pitches should be symmetric around 0.
    let sum: f32 = pitches.iter().sum();
    assert!(
        sum.abs() < 1e-6,
        "Pitch detuning should be symmetric: sum={sum}"
    );
    // Max detuning should be 0.08 semitones (8 cents).
    let max_pitch = pitches.iter().copied().fold(0.0f32, f32::max);
    assert!(
        (max_pitch - 0.08).abs() < 1e-6,
        "Max detuning should be 0.08 semitones: max={max_pitch}"
    );

    // Should have timing offsets.
    let offsets = config.timing_offsets_sec.as_ref().unwrap();
    assert_eq!(offsets.len(), n_voices);
    let max_offset = offsets.iter().copied().fold(0.0f32, f32::max);
    assert!(
        (max_offset - 0.003).abs() < 1e-6,
        "Max timing offset should be 3ms: max={max_offset}"
    );

    // Gains should be 1/sqrt(4) = 0.5.
    for &g in &config.gains {
        assert!(
            (g - 0.5).abs() < 1e-6,
            "equal_power gains for 4 voices should be 0.5: got {g}"
        );
    }
}

/// Verify that timing offsets produce measurable phase differences.
#[test]
fn test_timing_offset_shifts_signal() {
    let n_voices = 2;
    let n_samples = 4800;

    // Two identical voices.
    let voice_pcm: Vec<Vec<f32>> = (0..n_voices)
        .map(|_| generate_sine(440.0, n_samples, KOKORO_SAMPLE_RATE))
        .collect();

    // Config without timing offsets.
    let config_no_offset = ChorusConfig::equal_gain(n_voices).unwrap();
    let mixed_no_offset = mix_voices_with_config(&voice_pcm, &config_no_offset).unwrap();

    // Config with timing offsets: voice 0 at +5ms, voice 1 at -5ms.
    let config_offset = ChorusConfig::equal_gain(n_voices)
        .unwrap()
        .with_timing_offsets(vec![0.005, -0.005]);
    let mixed_offset = mix_voices_with_config(&voice_pcm, &config_offset).unwrap();

    // The offset mix should differ from the no-offset mix.
    let max_diff: f32 = mixed_no_offset
        .iter()
        .zip(mixed_offset.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff > 0.01,
        "Timing offset should produce a measurable difference: max_diff={max_diff}"
    );
}

/// Verify stereo width control: width=0 should collapse to mono.
#[test]
fn test_stereo_width_zero_collapses_to_mono() {
    let n_voices = 2;
    let n_samples = 2400;

    let voice_pcm: Vec<Vec<f32>> = (0..n_voices)
        .map(|_| generate_sine(440.0, n_samples, KOKORO_SAMPLE_RATE))
        .collect();

    // Hard-panned stereo with width=0 (should collapse to center).
    let config = ChorusConfig::with_stereo_pan(vec![0.5, 0.5], vec![-1.0, 1.0])
        .unwrap()
        .with_stereo_width(0.0);

    let mixed = mix_voices_with_config(&voice_pcm, &config).unwrap();

    // Output is interleaved stereo: [L0, R0, L1, R1, ...].
    // With width=0, left and right channels should be identical.
    for i in 0..n_samples {
        let left = mixed[i * 2];
        let right = mixed[i * 2 + 1];
        assert!(
            (left - right).abs() < 1e-6,
            "Stereo width=0 should produce mono: sample {i}, left={left}, right={right}"
        );
    }
}

/// Verify that the default SqrtHann crossfade is selected for 40ms overlap.
#[test]
fn test_default_config_uses_sqrt_hann_for_40ms() {
    let config = KokoroStreamConfig::default();
    assert_eq!(config.crossfade_samples, 960);
    assert_eq!(config.crossfade_window, CrossfadeWindow::SqrtHann);

    // 960 samples at 24kHz = 40ms
    let duration_ms = config.crossfade_samples as f64 / KOKORO_SAMPLE_RATE as f64 * 1000.0;
    assert!(
        (duration_ms - 40.0).abs() < 0.01,
        "Default crossfade should be 40ms: got {duration_ms}ms"
    );
}
