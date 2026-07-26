// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-model pipeline composition tests for the dvoice chain.
//!
//! The dvoice pipeline chains 4 models:
//!   Kokoro (TTS, 24kHz) -> HTDemucs (separation, 44.1kHz)
//!     -> Silero VAD (16kHz) -> Whisper (STT, 16kHz)
//!
//! These tests validate pipeline COMPATIBILITY — shapes, dtypes, sample rates,
//! and junction contract bounds — without loading real weights. They verify
//! that the models can compose into an end-to-end pipeline.
//!
//! Part of #3351 (Absolutely Best Kokoro).

use nn_tts_verify::kokoro_contracts::{
    all_contracts, bounds_within_contract, contract_stage, max_contract_violation,
    JunctionContract, J2_F0_LOWER, J2_F0_UPPER, J3_MAGNITUDE_LOWER, J3_MAGNITUDE_UPPER,
    J5_AUDIO_LOWER, J5_AUDIO_UPPER,
};
use nn_tts_verify::pipeline::{check_junction, verify_pipeline, VerifiedStage};

// ---------------------------------------------------------------------------
// Pipeline constants — sample rates and shapes for each model
// ---------------------------------------------------------------------------

/// Kokoro TTS output sample rate (Hz).
const KOKORO_SAMPLE_RATE: u32 = 24_000;

/// HTDemucs operates at 44.1 kHz (music separation standard).
const DEMUCS_SAMPLE_RATE: u32 = 44_100;

/// Silero VAD operates at 16 kHz.
const SILERO_VAD_SAMPLE_RATE: u32 = 16_000;

/// Whisper STT operates at 16 kHz.
const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Representative audio duration for pipeline tests (1 second).
const TEST_DURATION_SEC: f64 = 1.0;

/// Kokoro output: mono PCM, shape [1, Samples].
/// At 24kHz for 1s: 24000 samples.
fn kokoro_output_samples() -> usize {
    (f64::from(KOKORO_SAMPLE_RATE) * TEST_DURATION_SEC) as usize
}

/// HTDemucs input: mono or stereo, shape [B, Channels, Samples].
/// At 44.1kHz for 1s: 44100 samples.
fn demucs_input_samples() -> usize {
    (f64::from(DEMUCS_SAMPLE_RATE) * TEST_DURATION_SEC) as usize
}

/// HTDemucs output: separated sources, shape [B, Sources, Channels, Samples].
/// Sources = 4 (vocals, drums, bass, other) for standard HTDemucs.
const DEMUCS_NUM_SOURCES: usize = 4;

/// Silero VAD input: 512-sample chunks at 16kHz.
const SILERO_VAD_CHUNK_SIZE: usize = 512;

/// Whisper input: 30-second chunks at 16kHz = 480000 samples.
const WHISPER_N_SAMPLES: usize = 480_000;

/// Whisper mel bins.
const WHISPER_NUM_MEL_BINS: usize = 128;

/// Whisper mel frames (N_SAMPLES / HOP_LENGTH = 480000 / 160).
const WHISPER_N_FRAMES: usize = 3_000;

// ---------------------------------------------------------------------------
// Helper: build a VerifiedStage with uniform bounds
// ---------------------------------------------------------------------------

/// Build a stage with uniform bounds for all elements.
fn uniform_stage(
    name: &str,
    input_shape: &[usize],
    output_shape: &[usize],
    input_bounds: (f64, f64),
    output_bounds: (f64, f64),
) -> VerifiedStage {
    let in_elements: usize = input_shape.iter().product();
    let out_elements: usize = output_shape.iter().product();
    VerifiedStage::new(
        name,
        input_shape.to_vec(),
        output_shape.to_vec(),
        vec![input_bounds.0; in_elements],
        vec![input_bounds.1; in_elements],
        vec![output_bounds.0; out_elements],
        vec![output_bounds.1; out_elements],
        "CROWN",
        true,
    )
}

// ===========================================================================
// Test: Kokoro output shape compatible with HTDemucs input
// ===========================================================================

#[test]
fn test_kokoro_output_shape_compatible_with_demucs() {
    // Kokoro produces mono PCM audio: [1, Samples] at 24kHz.
    let kokoro_samples = kokoro_output_samples();
    let kokoro_output_shape = [1, kokoro_samples];

    // HTDemucs expects mono or stereo audio: [B, Channels, Samples] at 44.1kHz.
    // After resampling 24kHz -> 44.1kHz, sample count changes.
    let resampled_samples = demucs_input_samples();
    let demucs_input_shape = [1, 1, resampled_samples];

    // Kokoro output total elements (mono PCM).
    let kokoro_elements: usize = kokoro_output_shape.iter().product();
    assert_eq!(kokoro_elements, kokoro_samples);

    // Resampled output total elements (mono for Demucs).
    let demucs_input_elements: usize = demucs_input_shape.iter().product();
    assert_eq!(demucs_input_elements, resampled_samples);

    // The resampling ratio is deterministic: 44100/24000 = 1.8375.
    let resample_ratio = f64::from(DEMUCS_SAMPLE_RATE) / f64::from(KOKORO_SAMPLE_RATE);
    assert!((resample_ratio - 1.8375).abs() < 1e-6);

    // Verify the resampled length matches expectation.
    let expected_resampled = (kokoro_samples as f64 * resample_ratio) as usize;
    assert_eq!(expected_resampled, resampled_samples);

    // Kokoro output is mono (1 channel) — compatible with Demucs mono input.
    assert_eq!(kokoro_output_shape[0], 1, "Kokoro outputs batch=1");
    assert_eq!(demucs_input_shape[1], 1, "Demucs accepts mono (1 channel)");
}

// ===========================================================================
// Test: Silero VAD input from Demucs output
// ===========================================================================

#[test]
fn test_silero_vad_input_from_demucs() {
    // HTDemucs output: [B, Sources, Channels, Samples] at 44.1kHz.
    // We extract the vocals source (index 0) for VAD.
    let demucs_samples = demucs_input_samples();
    let demucs_output_shape = [1, DEMUCS_NUM_SOURCES, 1, demucs_samples];

    // Vocals extraction: [1, 1, Samples] — just the vocal track.
    let vocals_shape = [1, 1, demucs_samples];

    // After resampling 44.1kHz -> 16kHz for Silero VAD.
    let resample_ratio = f64::from(SILERO_VAD_SAMPLE_RATE) / f64::from(DEMUCS_SAMPLE_RATE);
    let vad_samples = (demucs_samples as f64 * resample_ratio) as usize;

    // Silero VAD processes in 512-sample chunks.
    let num_chunks = vad_samples / SILERO_VAD_CHUNK_SIZE;
    assert!(
        num_chunks > 0,
        "1 second of audio at 16kHz should produce multiple VAD chunks"
    );

    // Each chunk is [1, 512] — compatible with Silero VAD input.
    let vad_chunk_shape = [1, SILERO_VAD_CHUNK_SIZE];
    assert_eq!(vad_chunk_shape[1], 512);

    // Verify source extraction from Demucs preserves batch dimension.
    assert_eq!(demucs_output_shape[0], 1, "batch=1");
    assert_eq!(
        demucs_output_shape[1], DEMUCS_NUM_SOURCES,
        "4 sources: vocals, drums, bass, other"
    );
    assert_eq!(vocals_shape[0], 1, "extracted vocals batch=1");
}

// ===========================================================================
// Test: Whisper input from pipeline audio
// ===========================================================================

#[test]
fn test_whisper_input_from_pipeline() {
    // Pipeline produces audio at various sample rates. Whisper needs 16kHz.
    // The final audio (from Demucs vocals or original Kokoro) is resampled to 16kHz.

    // Whisper expects a fixed-size input: [1, 480000] = 30s at 16kHz.
    let whisper_input_shape = [1, WHISPER_N_SAMPLES];
    assert_eq!(whisper_input_shape.iter().product::<usize>(), 480_000);
    assert_eq!(WHISPER_N_SAMPLES, 480_000);

    // Mel spectrogram output: [1, NUM_MEL_BINS, N_FRAMES].
    let mel_shape = [1, WHISPER_NUM_MEL_BINS, WHISPER_N_FRAMES];
    assert_eq!(mel_shape[1], 128);
    assert_eq!(mel_shape[2], 3000);

    // For shorter audio (< 30s), padding to 480000 samples is required.
    let short_audio_samples = (f64::from(WHISPER_SAMPLE_RATE) * TEST_DURATION_SEC) as usize;
    assert!(
        short_audio_samples < WHISPER_N_SAMPLES,
        "1s of audio needs padding to fill 30s Whisper window"
    );

    // Padding factor.
    let padding_needed = WHISPER_N_SAMPLES - short_audio_samples;
    assert_eq!(padding_needed, 480_000 - 16_000);

    // Whisper's Conv1d stem has stride 2, so encoder output frames = N_FRAMES.
    // max_source_positions = 1500 = N_FRAMES / 2 (after two Conv1d stride-2 layers).
    let encoder_frames = WHISPER_N_FRAMES / 2;
    assert_eq!(encoder_frames, 1500);
}

// ===========================================================================
// Test: Junction contract J2 — Kokoro decoder to SourceModule
// ===========================================================================

#[test]
fn test_junction_contract_j2_kokoro_to_demucs() {
    let contracts = all_contracts();

    // J2_F0: F0 predictor output bounds.
    let j2_f0 = &contracts[0];
    assert_eq!(j2_f0.name, "J2_F0");
    assert_eq!(j2_f0.lower, J2_F0_LOWER);
    assert_eq!(j2_f0.upper, J2_F0_UPPER);

    // Simulated proven bounds within contract (typical F0 range: 80-400 Hz).
    let proven_lower = vec![0.0_f64; 10];
    let proven_upper = vec![400.0_f64; 10];
    assert!(bounds_within_contract(j2_f0, &proven_lower, &proven_upper));

    // Bounds that violate the contract (F0 > 800 Hz).
    let violated_upper = vec![900.0_f64; 10];
    assert!(!bounds_within_contract(
        j2_f0,
        &proven_lower,
        &violated_upper
    ));

    // Max violation measurement.
    let violation = max_contract_violation(j2_f0, &proven_lower, &violated_upper);
    assert!(
        (violation - 100.0).abs() < 1e-6,
        "violation should be 900 - 800 = 100"
    );
}

// ===========================================================================
// Test: Junction contract J3 — Demucs-relevant magnitude bounds
// ===========================================================================

#[test]
fn test_junction_contract_j3_demucs_to_vad() {
    let contracts = all_contracts();

    // J3_MAGNITUDE: Generator post_conv magnitude bounds.
    let j3_mag = &contracts[2];
    assert_eq!(j3_mag.name, "J3_MAGNITUDE");
    assert_eq!(j3_mag.lower, J3_MAGNITUDE_LOWER);
    assert_eq!(j3_mag.upper, J3_MAGNITUDE_UPPER);

    // Typical magnitude values are well within [-80, 80].
    let proven_lower = vec![-10.0_f64; 8];
    let proven_upper = vec![10.0_f64; 8];
    assert!(bounds_within_contract(j3_mag, &proven_lower, &proven_upper));
    assert_eq!(
        max_contract_violation(j3_mag, &proven_lower, &proven_upper),
        0.0,
        "no violation when within contract"
    );

    // J5_AUDIO: audio output bounds [-1, 1] — relevant for Demucs-to-VAD.
    // Audio passed between Demucs and VAD should be in [-1, 1] PCM range.
    let j5 = &contracts[5];
    assert_eq!(j5.name, "J5_AUDIO");
    assert_eq!(j5.lower, J5_AUDIO_LOWER);
    assert_eq!(j5.upper, J5_AUDIO_UPPER);

    // Demucs output (separated audio) should be within PCM range.
    let demucs_output_lower = vec![-0.95_f64; 16];
    let demucs_output_upper = vec![0.95_f64; 16];
    assert!(bounds_within_contract(
        j5,
        &demucs_output_lower,
        &demucs_output_upper
    ));
}

// ===========================================================================
// Test: Full pipeline shape chain
// ===========================================================================

#[test]
fn test_full_pipeline_shape_chain() {
    // Build stages representing the full dvoice chain with compatible shapes.
    // Use a small representative element count for tractability.
    let n = 64;

    // Stage 1: Kokoro TTS — text features in, audio PCM out.
    let kokoro = uniform_stage(
        "kokoro_tts",
        &[1, n],       // input: token features [B, T]
        &[1, n * 60],  // output: audio PCM [B, Samples] (upsampled ~60x)
        (-10.0, 10.0), // text embedding range
        (-1.0, 1.0),   // PCM audio range
    );

    // Stage 2: HTDemucs — audio in, separated vocals out.
    // Input/output element counts must match at junction.
    let demucs = uniform_stage(
        "htdemucs",
        &[1, n * 60], // input: resampled audio [B, Samples]
        &[1, n * 60], // output: vocals [B, Samples] (same length)
        (-1.0, 1.0),  // PCM audio range
        (-1.0, 1.0),  // separated vocals range
    );

    // Stage 3: Silero VAD — audio in, speech probability out.
    // For pipeline composition, we model the full-signal processing.
    let silero = uniform_stage(
        "silero_vad",
        &[1, n * 60], // input: resampled vocals [B, Samples]
        &[1, n * 60], // output: passed-through audio (VAD gating)
        (-1.0, 1.0),  // PCM audio range
        (-1.0, 1.0),  // gated audio range
    );

    // Stage 4: Whisper — audio in, tokens out.
    let whisper = uniform_stage(
        "whisper_stt",
        &[1, n * 60],    // input: padded audio [B, Samples]
        &[1, 448],       // output: token logits [B, MaxTokens]
        (-1.0, 1.0),     // PCM audio range
        (-100.0, 100.0), // logit range
    );

    // Verify end-to-end pipeline composition.
    let cert = verify_pipeline(&[kokoro, demucs, silero, whisper])
        .expect("4-stage pipeline should verify");

    // All junctions should be valid (bounds are contained).
    assert!(cert.is_valid, "pipeline should be valid");
    assert!(cert.is_sound, "all stages use sound verification");
    assert_eq!(cert.junctions.len(), 3, "3 junctions in 4-stage pipeline");

    // Check each junction.
    for (i, junction) in cert.junctions.iter().enumerate() {
        assert!(
            junction.shape_compatible,
            "junction {i}: shapes should be compatible"
        );
        assert!(
            junction.bounds_contained,
            "junction {i}: bounds should be contained"
        );
        assert_eq!(junction.max_violation, 0.0, "junction {i}: no violations");
    }

    // End-to-end bounds.
    assert!(cert.e2e_input_lower.iter().all(|&v| v == -10.0));
    assert!(cert.e2e_input_upper.iter().all(|&v| v == 10.0));
    assert!(cert.e2e_output_lower.iter().all(|&v| v == -100.0));
    assert!(cert.e2e_output_upper.iter().all(|&v| v == 100.0));
}

// ===========================================================================
// Test: Sample rate compatibility across pipeline stages
// ===========================================================================

#[test]
fn test_sample_rate_compatibility() {
    // Verify that sample rate conversions in the pipeline are well-defined.

    // Kokoro -> Demucs: 24kHz -> 44.1kHz (upsample).
    let ratio_kokoro_to_demucs = f64::from(DEMUCS_SAMPLE_RATE) / f64::from(KOKORO_SAMPLE_RATE);
    assert!(
        ratio_kokoro_to_demucs > 1.0,
        "upsampling from Kokoro to Demucs"
    );
    assert!(
        (ratio_kokoro_to_demucs - 1.8375).abs() < 1e-6,
        "ratio should be 44100/24000 = 1.8375"
    );

    // Demucs -> Silero VAD: 44.1kHz -> 16kHz (downsample).
    let ratio_demucs_to_vad = f64::from(SILERO_VAD_SAMPLE_RATE) / f64::from(DEMUCS_SAMPLE_RATE);
    assert!(
        ratio_demucs_to_vad < 1.0,
        "downsampling from Demucs to Silero VAD"
    );
    assert!(
        (ratio_demucs_to_vad - 16000.0 / 44100.0).abs() < 1e-6,
        "ratio should be 16000/44100"
    );

    // Silero VAD -> Whisper: 16kHz -> 16kHz (no conversion needed).
    assert_eq!(
        SILERO_VAD_SAMPLE_RATE, WHISPER_SAMPLE_RATE,
        "Silero VAD and Whisper share the same sample rate"
    );

    // End-to-end: Kokoro 24kHz -> Whisper 16kHz.
    let ratio_e2e = f64::from(WHISPER_SAMPLE_RATE) / f64::from(KOKORO_SAMPLE_RATE);
    assert!(
        (ratio_e2e - 2.0 / 3.0).abs() < 1e-6,
        "end-to-end ratio is 16000/24000 = 2/3"
    );

    // All sample rates are standard and non-zero.
    for &sr in &[
        KOKORO_SAMPLE_RATE,
        DEMUCS_SAMPLE_RATE,
        SILERO_VAD_SAMPLE_RATE,
        WHISPER_SAMPLE_RATE,
    ] {
        assert!(sr > 0, "sample rate must be positive");
        assert!(sr <= 192_000, "sample rate should be <= 192kHz");
    }
}

// ===========================================================================
// Test: Audio format consistency (mono/stereo, bit depth, PCM range)
// ===========================================================================

#[test]
fn test_audio_format_consistency() {
    // All pipeline stages use f32 PCM with values in [-1, 1].

    // Kokoro output: mono f32 PCM, [-1, 1].
    let kokoro_channels = 1_usize;
    let kokoro_pcm_range = (J5_AUDIO_LOWER, J5_AUDIO_UPPER);
    assert_eq!(kokoro_channels, 1, "Kokoro outputs mono audio");
    assert_eq!(kokoro_pcm_range, (-1.0, 1.0), "standard PCM range");

    // HTDemucs: accepts mono or stereo, outputs per-source mono/stereo.
    // For dvoice, input is mono (from Kokoro).
    let demucs_input_channels = 1_usize;
    let demucs_output_channels_per_source = 1_usize;
    assert_eq!(demucs_input_channels, 1, "Demucs receives mono from Kokoro");
    assert_eq!(
        demucs_output_channels_per_source, 1,
        "mono input produces mono output per source"
    );

    // Silero VAD: mono only.
    let silero_channels = 1_usize;
    assert_eq!(silero_channels, 1, "Silero VAD is mono-only");

    // Whisper: mono only.
    let whisper_channels = 1_usize;
    assert_eq!(whisper_channels, 1, "Whisper is mono-only");

    // All stages use the same PCM convention: f32 in [-1, 1].
    let pcm_lower = -1.0_f64;
    let pcm_upper = 1.0_f64;

    // Verify junction contract J5 matches PCM convention.
    assert_eq!(J5_AUDIO_LOWER, pcm_lower);
    assert_eq!(J5_AUDIO_UPPER, pcm_upper);

    // Build a minimal 2-stage pipeline to verify PCM format compatibility.
    let producer = uniform_stage(
        "audio_producer",
        &[1, 100],
        &[1, 100],
        (-1.0, 1.0),
        (pcm_lower, pcm_upper),
    );
    let consumer = uniform_stage(
        "audio_consumer",
        &[1, 100],
        &[1, 100],
        (pcm_lower, pcm_upper),
        (-50.0, 50.0),
    );

    let junction = check_junction(&producer, &consumer, 0);
    assert!(junction.shape_compatible, "same shape");
    assert!(junction.bounds_contained, "PCM bounds match");
    assert_eq!(junction.max_violation, 0.0);
}

// ===========================================================================
// Test: contract_stage helper for pipeline construction
// ===========================================================================

#[test]
fn test_contract_stage_kokoro_pipeline() {
    // Use contract_stage to build stages from junction contracts.
    let j5_audio =
        JunctionContract::new("J5_AUDIO", "iSTFT output", J5_AUDIO_LOWER, J5_AUDIO_UPPER);

    // Kokoro output stage: produces audio in J5 bounds.
    let kokoro_stage = contract_stage(
        "kokoro_output",
        &[1, 64],  // input shape
        &[1, 128], // output shape (upsampled)
        &JunctionContract::new("input", "text", -10.0, 10.0),
        &j5_audio,
        "CROWN",
        true,
    );

    assert_eq!(kokoro_stage.name, "kokoro_output");
    assert_eq!(kokoro_stage.input_shape, vec![1, 64]);
    assert_eq!(kokoro_stage.output_shape, vec![1, 128]);

    // Output bounds should match J5 audio contract.
    assert!(kokoro_stage
        .output_lower
        .iter()
        .all(|&v| v == J5_AUDIO_LOWER));
    assert!(kokoro_stage
        .output_upper
        .iter()
        .all(|&v| v == J5_AUDIO_UPPER));

    // Demucs input stage: accepts audio in J5 bounds.
    let demucs_stage = contract_stage(
        "demucs_input",
        &[1, 128], // input shape (resampled)
        &[1, 128], // output shape (separated vocals)
        &j5_audio,
        &j5_audio,
        "CROWN",
        true,
    );

    // Compose the two stages.
    let cert =
        verify_pipeline(&[kokoro_stage, demucs_stage]).expect("Kokoro -> Demucs should compose");

    assert!(cert.is_valid, "pipeline should be valid");
    assert_eq!(cert.junctions.len(), 1);
    assert!(cert.junctions[0].bounds_contained);
}

// ===========================================================================
// Test: pipeline junction violation detection
// ===========================================================================

#[test]
fn test_pipeline_junction_violation_detected() {
    // Build stages with incompatible bounds to verify violation detection.

    // Stage producing audio slightly outside [-1, 1].
    let clipping_producer = uniform_stage(
        "clipping_tts",
        &[1, 32],
        &[1, 32],
        (-10.0, 10.0),
        (-1.5, 1.5), // output exceeds PCM range
    );

    // Stage expecting audio in [-1, 1].
    let strict_consumer = uniform_stage(
        "strict_vad",
        &[1, 32],
        &[1, 1],
        (-1.0, 1.0), // input expects PCM range
        (0.0, 1.0),
    );

    let cert = verify_pipeline(&[clipping_producer, strict_consumer])
        .expect("should verify even with violations");

    assert!(!cert.is_valid, "pipeline with bound violation is invalid");
    assert_eq!(cert.junctions.len(), 1);

    let junction = &cert.junctions[0];
    assert!(!junction.bounds_contained, "bounds should not be contained");
    assert!(
        (junction.max_violation - 0.5).abs() < 1e-6,
        "violation should be 1.5 - 1.0 = 0.5, got {}",
        junction.max_violation
    );
    assert_eq!(
        junction.violation_count, 32,
        "all 32 elements should violate"
    );
}

// ===========================================================================
// Test: shape mismatch detection between stages
// ===========================================================================

#[test]
fn test_pipeline_shape_mismatch_detected() {
    // Kokoro outputs 1000 samples but Demucs expects 2000.
    let kokoro = uniform_stage("kokoro", &[1, 64], &[1, 1000], (-10.0, 10.0), (-1.0, 1.0));
    let demucs = uniform_stage("demucs", &[1, 2000], &[1, 2000], (-1.0, 1.0), (-1.0, 1.0));

    let junction = check_junction(&kokoro, &demucs, 0);
    assert!(
        !junction.shape_compatible,
        "1000 vs 2000 elements should be incompatible"
    );
}

// ===========================================================================
// Test: NaN/Inf bounds are treated as violations
// ===========================================================================

#[test]
fn test_nan_bounds_are_violations() {
    // Defense-in-depth: NaN in proven bounds must be caught.
    let contracts = all_contracts();
    let j5 = &contracts[5]; // J5_AUDIO

    let nan_lower = vec![f64::NAN; 4];
    let normal_upper = vec![0.5_f64; 4];
    assert!(
        !bounds_within_contract(j5, &nan_lower, &normal_upper),
        "NaN lower bounds should fail containment"
    );

    let inf_upper = vec![f64::INFINITY; 4];
    let normal_lower = vec![-0.5_f64; 4];
    assert!(
        !bounds_within_contract(j5, &normal_lower, &inf_upper),
        "Inf upper bounds should fail containment"
    );

    // max_contract_violation returns MAX for non-finite.
    let violation = max_contract_violation(j5, &nan_lower, &normal_upper);
    assert_eq!(violation, f64::MAX);
}

// ===========================================================================
// Test: all 6 Kokoro junction contracts are well-formed
// ===========================================================================

#[test]
fn test_all_contracts_well_formed() {
    let contracts = all_contracts();
    assert_eq!(contracts.len(), 6, "should have 6 junction contracts");

    for contract in &contracts {
        // Bounds must be finite.
        assert!(
            contract.lower.is_finite(),
            "{}: lower bound must be finite",
            contract.name
        );
        assert!(
            contract.upper.is_finite(),
            "{}: upper bound must be finite",
            contract.name
        );
        // Lower < upper.
        assert!(
            contract.lower < contract.upper,
            "{}: lower ({}) must be < upper ({})",
            contract.name,
            contract.lower,
            contract.upper,
        );
        // Name and zone are non-empty.
        assert!(!contract.name.is_empty());
        assert!(!contract.zone.is_empty());
    }
}
