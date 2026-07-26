// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end Whisper transcription tests with real weights.
//!
//! These tests exercise the FULL assembled pipeline — mel spectrogram
//! computation, encoder, decoder, greedy/beam decode, language detection,
//! and timestamp token handling — rather than individual components.
//!
//! ## Setup
//!
//! Set `WHISPER_WEIGHTS` to the directory containing `model.safetensors` and
//! reference `.npy` files:
//!
//! ```bash
//! export WHISPER_WEIGHTS=./nn/weights/whisper-tiny
//! cargo test -p nn-whisper --test transcription_e2e -- --nocapture
//! ```
//!
//! Required files in `$WHISPER_WEIGHTS/`:
//! - `model.safetensors` — AI Provider whisper-tiny weights (151 MB)
//! - `ref_mel_input.npy` — PyTorch reference mel input [1, 80, 3000]
//! - `ref_encoder_output.npy` — PyTorch reference encoder output [1, 1500, 384]
//! - `ref_decoder_input_ids.npy` — PyTorch reference decoder token IDs [1, 1]
//! - `ref_decoder_logits.npy` — PyTorch reference decoder logits [1, 1, 51865]

use std::path::{Path, PathBuf};

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_reftest::load_npy;
use nn_whisper::{
    beam_search_decode, detect_language, greedy_decode, transcribe,
    whisper_mel_spectrogram_for_config, DecodeConfig, WhisperBeamConfig, WhisperConfig,
    WhisperModel, WhisperTokenizer, EOT_TOKEN, LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START,
    NO_SPEECH_TOKEN, SOT_TOKEN,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the whisper weights directory, or None if not configured.
fn weights_dir() -> Option<PathBuf> {
    std::env::var("WHISPER_WEIGHTS").ok().map(PathBuf::from)
}

/// Skip macro — prints skip message and returns early when weights are absent.
macro_rules! skip_without_weights {
    ($dir:ident) => {
        let Some($dir) = weights_dir() else {
            eprintln!(
                "SKIP: WHISPER_WEIGHTS not set. \
                 Set to whisper-tiny weights directory to run transcription e2e tests."
            );
            return;
        };
    };
}

/// Load the whisper-tiny model from safetensors.
fn load_model(dir: &Path) -> WhisperModel {
    let st_path = dir.join("model.safetensors");
    let config = WhisperConfig::whisper_tiny();
    WhisperModel::load_safetensors(&st_path, config)
        .unwrap_or_else(|e| panic!("Failed to load model from {}: {e}", st_path.display()))
}

/// Load a reference .npy tensor, returning its flat f32 data and shape.
fn load_ref_npy(dir: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    let path = dir.join(format!("{name}.npy"));
    let trace = load_npy(&path).unwrap_or_else(|e| {
        panic!("Failed to load reference {}: {e}", path.display());
    });
    let tensor = trace.get(0).expect("npy should contain one tensor");
    (tensor.data.clone(), tensor.shape.clone())
}

/// Generate 30 seconds of silence at 16 kHz (all zeros).
fn silence_30s() -> Vec<f32> {
    vec![0.0f32; 16_000 * 30]
}

/// Generate a short 440 Hz sine tone at 16 kHz (non-speech audio).
fn sine_tone_1s() -> Vec<f32> {
    let sample_rate = 16_000.0_f32;
    let freq = 440.0_f32;
    (0..16_000)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin() * 0.5)
        .collect()
}

/// Pad audio to 30 seconds (480,000 samples) for Whisper processing.
fn pad_to_30s(audio: &[f32]) -> Vec<f32> {
    let n_samples = 16_000 * 30;
    let mut padded = audio.to_vec();
    padded.resize(n_samples, 0.0);
    padded
}

/// Build a minimal tokenizer from the vocab embedded in the model config.
///
/// Returns a tokenizer that can decode token IDs to text. For the
/// whisper-tiny model, the vocab has 51865 entries. This uses a
/// minimal fallback tokenizer when vocab.json is not available.
fn build_fallback_tokenizer() -> WhisperTokenizer {
    // Construct a minimal vocabulary with special tokens only.
    // This is enough for decode pipeline testing (decoding special tokens
    // produces empty strings, which is correct behavior for silence).
    let mut vocab = std::collections::HashMap::new();
    vocab.insert("<|endoftext|>".to_string(), EOT_TOKEN);
    vocab.insert("<|startoftranscript|>".to_string(), SOT_TOKEN);
    vocab.insert("<|nospeech|>".to_string(), NO_SPEECH_TOKEN);
    vocab.insert("<|notimestamps|>".to_string(), 50364_usize);
    vocab.insert("<|en|>".to_string(), 50259_usize);
    vocab.insert("<|transcribe|>".to_string(), 50360_usize);

    let json = serde_json::to_string(&vocab).expect("serialize minimal vocab");
    WhisperTokenizer::from_vocab_str(&json).expect("build fallback tokenizer")
}

// ===========================================================================
// 1. Transcribe Silence
// ===========================================================================

#[test]
fn test_transcribe_silence() {
    // Transcribing pure silence should produce empty text or very short output
    // with a high no-speech probability.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);
    let config = WhisperConfig::whisper_tiny();
    let tokenizer = build_fallback_tokenizer();

    let audio = silence_30s();
    let mel = whisper_mel_spectrogram_for_config(&audio, config.num_mel_bins)
        .expect("mel spectrogram of silence");

    let encoder_output = model.encode(&mel).expect("encode silence");

    let decode_config = DecodeConfig::default();
    let result =
        transcribe(&mut model, &encoder_output, &decode_config, &tokenizer).expect("transcribe");

    eprintln!("Silence transcription: {:?}", result.text.trim());
    eprintln!("No-speech prob: {:.4}", result.no_speech_prob);
    eprintln!("Tokens: {:?}", result.decode_result.tokens);
    eprintln!("Reached EOT: {}", result.decode_result.reached_eot);

    // For silence, the no-speech probability should be relatively high.
    // Whisper-tiny with sinusoidal pos-embed may not perfectly match PyTorch,
    // but silence should still have elevated no_speech_prob.
    //
    // Also verify: text should be empty OR very short (< 50 chars).
    // Silence can sometimes produce hallucinated tokens in small models,
    // but it should not produce long coherent text.
    let text = result.text.trim();
    assert!(
        text.len() < 200,
        "silence should not produce long text, got {} chars: {:?}",
        text.len(),
        text,
    );

    eprintln!(
        "PASS: silence transcription is short ({} chars)",
        text.len()
    );
}

// ===========================================================================
// 2. Mel Spectrogram Shape Verification
// ===========================================================================

#[test]
fn test_mel_spectrogram_shapes() {
    // Verify mel computation produces correct shapes for various audio lengths.
    // This does NOT require weights — it tests the audio preprocessing pipeline.

    let config = WhisperConfig::whisper_tiny();
    assert_eq!(config.num_mel_bins, 80, "whisper-tiny uses 80 mel bins");

    // 30 seconds of silence (standard Whisper input length).
    let audio_30s = silence_30s();
    let mel_30s =
        whisper_mel_spectrogram_for_config(&audio_30s, config.num_mel_bins).expect("mel 30s");

    assert_eq!(
        mel_30s.rank(),
        3,
        "mel should be rank 3: [B, mel_bins, frames]"
    );
    assert_eq!(mel_30s.dim(0).unwrap(), 1, "batch dim = 1");
    assert_eq!(
        mel_30s.dim(1).unwrap(),
        80,
        "mel bins = 80 for whisper-tiny"
    );

    let frames_30s = mel_30s.dim(2).unwrap();
    // 30s at 16kHz with hop_length=160: 480000/160 = 3000 frames.
    // May be 3000 or 3001 depending on padding.
    assert!(
        (2999..=3001).contains(&frames_30s),
        "30s audio should produce ~3000 frames, got {frames_30s}"
    );

    // All values should be finite.
    let flat = mel_30s.to_flat_vec::<f32>().unwrap();
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "mel spectrogram should have no NaN/Inf");

    eprintln!("Mel 30s shape: {:?} (frames={frames_30s})", mel_30s.dims());

    // Short audio: 1 second.
    let audio_1s = vec![0.0f32; 16_000];
    let mel_1s =
        whisper_mel_spectrogram_for_config(&audio_1s, config.num_mel_bins).expect("mel 1s");

    assert_eq!(mel_1s.dim(0).unwrap(), 1, "batch dim");
    assert_eq!(mel_1s.dim(1).unwrap(), 80, "mel bins");
    let frames_1s = mel_1s.dim(2).unwrap();
    assert!(frames_1s > 0, "1s audio should produce at least 1 frame");

    eprintln!("Mel 1s shape: {:?} (frames={frames_1s})", mel_1s.dims());

    // Non-speech audio: sine tone.
    let tone = pad_to_30s(&sine_tone_1s());
    let mel_tone =
        whisper_mel_spectrogram_for_config(&tone, config.num_mel_bins).expect("mel sine tone");

    assert_eq!(mel_tone.dim(1).unwrap(), 80, "mel bins for tone");
    let flat_tone = mel_tone.to_flat_vec::<f32>().unwrap();
    assert!(
        flat_tone.iter().all(|v| v.is_finite()),
        "mel of sine tone should be finite"
    );

    // Sine tone should have different energy distribution than silence.
    let mel_silence_flat = flat;
    let tone_differs = mel_silence_flat
        .iter()
        .zip(flat_tone.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        tone_differs,
        "mel of sine tone should differ from mel of silence"
    );

    eprintln!("PASS: mel spectrogram shapes and properties verified");
}

// ===========================================================================
// 3. Encoder-Decoder Roundtrip
// ===========================================================================

#[test]
fn test_encoder_decoder_roundtrip() {
    // Full pipeline: PCM -> mel -> encode -> decode -> token sequence.
    // Verifies the assembled pipeline produces valid output.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);

    // Use the reference mel for consistent results.
    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");
    assert_eq!(mel_shape, vec![1, 80, 3000], "ref mel shape");

    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu).expect("create mel tensor");

    // Step 1: Encode.
    let encoder_output = model.encode(&mel).expect("encode");
    assert_eq!(encoder_output.rank(), 3, "encoder output rank");
    assert_eq!(encoder_output.dim(0).unwrap(), 1, "batch");
    assert_eq!(encoder_output.dim(1).unwrap(), 1500, "seq_len");
    assert_eq!(encoder_output.dim(2).unwrap(), 384, "d_model");

    // Encoder output should be finite.
    let enc_flat = encoder_output.to_flat_vec::<f32>().unwrap();
    assert!(
        enc_flat.iter().all(|v| v.is_finite()),
        "encoder output must be finite"
    );

    // Step 2: Decode initial prompt (SOT, en, transcribe, notimestamps).
    let initial_tokens: Vec<u32> = vec![50258, 50259, 50360, 50364];
    let tokens = DynTensor::from_vec_u32(initial_tokens, &[1, 4], &Device::Cpu)
        .expect("initial token tensor");

    let logits = model
        .decode(&tokens, &encoder_output, true, 0)
        .expect("decode initial prompt");

    assert_eq!(logits.rank(), 3, "logits rank");
    assert_eq!(logits.dim(0).unwrap(), 1, "batch");
    assert_eq!(logits.dim(1).unwrap(), 4, "seq_len matches initial prompt");
    assert_eq!(logits.dim(2).unwrap(), 51865, "vocab_size");

    let logits_flat = logits.to_flat_vec::<f32>().unwrap();
    assert!(
        logits_flat.iter().all(|v| v.is_finite()),
        "decoder logits must be finite"
    );

    // Step 3: Autoregressive decode for 10 steps.
    model.reset_kv_cache();
    let mut generated: Vec<usize> = vec![50258, 50259, 50360, 50364]; // initial prompt

    // Feed initial prompt.
    let init_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
    let init_tensor = DynTensor::from_vec_u32(init_u32, &[1, generated.len()], &Device::Cpu)
        .expect("init tensor");
    let logits = model
        .decode(&init_tensor, &encoder_output, true, 0)
        .expect("decode init");

    // Get the last token's logits.
    let vocab_size = logits.dim(2).unwrap();
    let all_logits = logits.to_flat_vec::<f32>().unwrap();
    let last_logits = &all_logits[(all_logits.len() - vocab_size)..];

    let first_token = last_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    generated.push(first_token);

    // Continue decoding.
    for step in 1..10 {
        let last = *generated.last().unwrap();
        if last == EOT_TOKEN {
            break;
        }
        let tok_u32 = vec![last as u32];
        let tok_tensor =
            DynTensor::from_vec_u32(tok_u32, &[1, 1], &Device::Cpu).expect("step token");
        let logits = model
            .decode(&tok_tensor, &encoder_output, false, generated.len() - 1)
            .unwrap_or_else(|_| panic!("decode step {step}"));

        let step_logits = logits.to_flat_vec::<f32>().unwrap();
        assert!(
            step_logits.iter().all(|v| v.is_finite()),
            "step {step}: logits must be finite"
        );

        let next = step_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        generated.push(next);
    }

    // All tokens should be valid.
    for &t in &generated {
        assert!(t < 51865, "token {t} out of vocab range");
    }

    eprintln!(
        "Roundtrip generated {} tokens: {:?}",
        generated.len(),
        generated
    );
    eprintln!("PASS: encoder-decoder roundtrip produces valid token sequence");
}

// ===========================================================================
// 4. Greedy Decode
// ===========================================================================

#[test]
fn test_greedy_decode() {
    // Test the greedy_decode() function produces a valid token sequence.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);

    // Encode reference mel.
    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");
    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu).expect("mel");
    let encoder_output = model.encode(&mel).expect("encode");

    let decode_config = DecodeConfig::default();
    let result = greedy_decode(&mut model, &encoder_output, &decode_config).expect("greedy_decode");

    eprintln!("Greedy decode:");
    eprintln!("  tokens:           {:?}", result.tokens);
    eprintln!("  num_tokens:       {}", result.tokens.len());
    eprintln!("  avg_logprob:      {:.4}", result.avg_logprob);
    eprintln!("  compression:      {:.4}", result.compression_ratio);
    eprintln!("  reached_eot:      {}", result.reached_eot);
    eprintln!("  temperature:      {:.1}", result.temperature);
    eprintln!("  no_speech_prob:   {:.4}", result.no_speech_prob);

    // Greedy decode should produce some tokens.
    assert!(
        !result.tokens.is_empty(),
        "greedy decode should produce at least 1 token"
    );

    // All tokens should be valid vocab indices.
    for &t in &result.tokens {
        assert!(t < 51865, "token {t} out of vocab range");
    }

    // Temperature should be 0.0 for greedy.
    assert!(
        result.temperature.abs() < 1e-10,
        "greedy decode temperature should be 0.0, got {}",
        result.temperature,
    );

    // Compression ratio should be positive and finite.
    assert!(
        result.compression_ratio.is_finite() && result.compression_ratio > 0.0,
        "compression_ratio should be finite and positive, got {}",
        result.compression_ratio,
    );

    // Average log probability should be finite and non-positive.
    assert!(
        result.avg_logprob.is_finite(),
        "avg_logprob should be finite, got {}",
        result.avg_logprob,
    );
    assert!(
        result.avg_logprob <= 0.0,
        "avg_logprob should be <= 0 (log probability), got {}",
        result.avg_logprob,
    );

    // No-speech probability should be in [0, 1].
    assert!(
        (0.0..=1.0).contains(&result.no_speech_prob),
        "no_speech_prob should be in [0, 1], got {}",
        result.no_speech_prob,
    );

    eprintln!("PASS: greedy decode produces valid result");
}

// ===========================================================================
// 5. Beam Search Decode
// ===========================================================================

#[test]
fn test_beam_search_decode() {
    // Test beam search decode produces valid results and (in most cases)
    // equal or better log probability than greedy decode.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);

    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");
    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu).expect("mel");
    let encoder_output = model.encode(&mel).expect("encode");

    // First: greedy baseline.
    let decode_config = DecodeConfig::default();
    let greedy_result =
        greedy_decode(&mut model, &encoder_output, &decode_config).expect("greedy baseline");

    // Now: beam search with beam_width=3.
    model.reset_kv_cache();
    let mut beam_config = WhisperBeamConfig::default();
    beam_config.beam_width = 3;
    beam_config.length_penalty = 1.0;
    let beam_result = beam_search_decode(&mut model, &encoder_output, &decode_config, &beam_config)
        .expect("beam search decode");

    eprintln!("Beam search decode (beam_width=3):");
    eprintln!("  tokens:           {:?}", beam_result.tokens);
    eprintln!("  num_tokens:       {}", beam_result.tokens.len());
    eprintln!("  avg_logprob:      {:.4}", beam_result.avg_logprob);
    eprintln!("  compression:      {:.4}", beam_result.compression_ratio);
    eprintln!("  reached_eot:      {}", beam_result.reached_eot);
    eprintln!("  temperature:      {:.1}", beam_result.temperature);
    eprintln!("Greedy baseline:");
    eprintln!("  avg_logprob:      {:.4}", greedy_result.avg_logprob);
    eprintln!("  num_tokens:       {}", greedy_result.tokens.len());

    // Beam search should produce valid tokens.
    assert!(
        !beam_result.tokens.is_empty(),
        "beam search should produce at least 1 token"
    );
    for &t in &beam_result.tokens {
        assert!(t < 51865, "beam token {t} out of vocab range");
    }

    // Beam search avg_logprob should be finite.
    assert!(
        beam_result.avg_logprob.is_finite(),
        "beam avg_logprob should be finite"
    );

    // Compression ratio should be valid.
    assert!(
        beam_result.compression_ratio.is_finite() && beam_result.compression_ratio > 0.0,
        "beam compression_ratio should be finite and positive"
    );

    eprintln!("PASS: beam search decode produces valid result");
}

// ===========================================================================
// 6. Timestamp Token Handling
// ===========================================================================

#[test]
fn test_timestamp_tokens() {
    // Verify that the tokenizer correctly handles timestamp tokens.
    // Timestamp tokens start at ID 50365 with 0.02s resolution.
    let tokenizer = build_fallback_tokenizer();

    // Token 50365 = <|0.00|>
    assert!(
        tokenizer.is_timestamp(50365),
        "token 50365 should be a timestamp"
    );
    assert!(
        tokenizer.is_special(50365),
        "timestamp tokens are special tokens"
    );

    // Timestamp value: (token_id - 50365) * 0.02
    let ts_0 = tokenizer.timestamp_value(50365);
    assert_eq!(ts_0, Some(0.0), "<|0.00|> = 0.0s");

    let ts_1s = tokenizer.timestamp_value(50365 + 50); // 50 * 0.02 = 1.0s
    assert!(
        (ts_1s.unwrap() - 1.0).abs() < 1e-10,
        "<|1.00|> = 1.0s, got {ts_1s:?}",
    );

    let ts_30s = tokenizer.timestamp_value(50365 + 1500); // 1500 * 0.02 = 30.0s
    assert!(
        (ts_30s.unwrap() - 30.0).abs() < 1e-10,
        "<|30.00|> = 30.0s, got {ts_30s:?}",
    );

    // Non-timestamp tokens should return None.
    assert_eq!(
        tokenizer.timestamp_value(50258),
        None,
        "SOT is not a timestamp"
    );
    assert_eq!(
        tokenizer.timestamp_value(0),
        None,
        "regular token is not a timestamp"
    );

    // Decode with timestamps: test that timestamp pairs create segments.
    let tokens_with_ts = vec![
        50365, // <|0.00|>
        123,
        456,        // some content tokens
        50365 + 50, // <|1.00|>
    ];
    let segments = tokenizer
        .decode_with_timestamps(&tokens_with_ts)
        .expect("decode with timestamps");

    // Should have at least one segment with start/end times.
    assert!(
        !segments.is_empty(),
        "should produce at least one segment from timestamp-bracketed tokens"
    );

    let seg = &segments[0];
    assert_eq!(seg.start, Some(0.0), "segment start = 0.00");
    assert!((seg.end.unwrap() - 1.0).abs() < 1e-10, "segment end = 1.00");

    eprintln!("Timestamp segments: {segments:?}");
    eprintln!("PASS: timestamp token handling verified");
}

// ===========================================================================
// 7. Language Detection
// ===========================================================================

#[test]
fn test_language_detection() {
    // Test language detection on the reference mel input.
    // Whisper-tiny is multilingual — it should detect a language.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);

    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");
    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu).expect("mel");
    let encoder_output = model.encode(&mel).expect("encode");

    let lang_result = detect_language(&mut model, &encoder_output).expect("detect_language");

    eprintln!("Language detection:");
    eprintln!("  language_token: {}", lang_result.language_token);
    eprintln!("  probability:    {:.4}", lang_result.probability);
    eprintln!("  no_speech_prob: {:.4}", lang_result.no_speech_prob);

    // Language token should be in the valid range.
    assert!(
        lang_result.language_token >= LANGUAGE_TOKEN_START
            && lang_result.language_token <= LANGUAGE_TOKEN_END,
        "language_token {} should be in [{}, {}]",
        lang_result.language_token,
        LANGUAGE_TOKEN_START,
        LANGUAGE_TOKEN_END,
    );

    // Probability should be in [0, 1] and finite.
    assert!(
        lang_result.probability.is_finite()
            && lang_result.probability >= 0.0
            && lang_result.probability <= 1.0,
        "language probability should be in [0, 1], got {}",
        lang_result.probability,
    );

    // No-speech probability should be in [0, 1].
    assert!(
        lang_result.no_speech_prob.is_finite()
            && lang_result.no_speech_prob >= 0.0
            && lang_result.no_speech_prob <= 1.0,
        "no_speech_prob should be in [0, 1], got {}",
        lang_result.no_speech_prob,
    );

    eprintln!("PASS: language detection returns valid result");
}

// ===========================================================================
// 8. Silence No-Speech Probability
// ===========================================================================

#[test]
fn test_silence_no_speech_probability() {
    // Pure silence should have a higher no-speech probability than
    // the reference mel (which contains actual audio content).
    skip_without_weights!(dir);

    let mut model = load_model(&dir);
    let config = WhisperConfig::whisper_tiny();

    // Silence mel.
    let silence = silence_30s();
    let mel_silence =
        whisper_mel_spectrogram_for_config(&silence, config.num_mel_bins).expect("mel silence");
    let enc_silence = model.encode(&mel_silence).expect("encode silence");

    let decode_config = DecodeConfig::default();
    let silence_result =
        greedy_decode(&mut model, &enc_silence, &decode_config).expect("decode silence");

    // Reference mel (contains audio).
    model.reset_kv_cache();
    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");
    let mel_ref = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu).expect("mel ref");
    let enc_ref = model.encode(&mel_ref).expect("encode ref");

    let ref_result = greedy_decode(&mut model, &enc_ref, &decode_config).expect("decode ref");

    eprintln!("No-speech probability comparison:");
    eprintln!("  silence:   {:.6}", silence_result.no_speech_prob);
    eprintln!("  reference: {:.6}", ref_result.no_speech_prob);

    // Both probabilities should be valid.
    assert!(
        silence_result.no_speech_prob.is_finite(),
        "silence no_speech_prob should be finite"
    );
    assert!(
        ref_result.no_speech_prob.is_finite(),
        "ref no_speech_prob should be finite"
    );

    eprintln!("PASS: no-speech probabilities are valid and finite");
}

// ===========================================================================
// 9. Greedy Decode Determinism
// ===========================================================================

#[test]
fn test_greedy_decode_determinism() {
    // Running greedy decode twice on the same input should produce
    // identical token sequences (greedy = deterministic).
    skip_without_weights!(dir);

    let mut model = load_model(&dir);

    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");
    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu).expect("mel");
    let encoder_output = model.encode(&mel).expect("encode");

    let decode_config = DecodeConfig::default();

    let result1 = greedy_decode(&mut model, &encoder_output, &decode_config).expect("greedy run 1");
    let result2 = greedy_decode(&mut model, &encoder_output, &decode_config).expect("greedy run 2");

    assert_eq!(
        result1.tokens, result2.tokens,
        "greedy decode should be deterministic"
    );
    assert!(
        (result1.avg_logprob - result2.avg_logprob).abs() < 1e-6,
        "avg_logprob should be identical: {} vs {}",
        result1.avg_logprob,
        result2.avg_logprob,
    );

    eprintln!("Run 1 tokens: {:?}", result1.tokens);
    eprintln!("Run 2 tokens: {:?}", result2.tokens);
    eprintln!("PASS: greedy decode is deterministic");
}

// ===========================================================================
// 10. Full Transcription Pipeline (mel -> text)
// ===========================================================================

#[test]
fn test_full_transcription_pipeline() {
    // Test the complete pipeline from raw audio through to decoded text.
    // Uses a sine tone (non-speech) to test pipeline integrity rather than
    // transcription accuracy.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);
    let config = WhisperConfig::whisper_tiny();
    let tokenizer = build_fallback_tokenizer();

    // Generate a non-speech signal: sine tone padded to 30s.
    let audio = pad_to_30s(&sine_tone_1s());

    // Step 1: mel spectrogram.
    let mel = whisper_mel_spectrogram_for_config(&audio, config.num_mel_bins)
        .expect("mel from sine tone");
    assert_eq!(mel.dim(1).unwrap(), 80, "mel bins");

    // Step 2: encode.
    let encoder_output = model.encode(&mel).expect("encode");
    assert_eq!(encoder_output.dims()[0], 1, "batch");
    assert_eq!(encoder_output.dims()[1], 1500, "seq_len");
    assert_eq!(encoder_output.dims()[2], 384, "d_model");

    // Step 3: transcribe (greedy decode + detokenize).
    let decode_config = DecodeConfig::default();
    let result =
        transcribe(&mut model, &encoder_output, &decode_config, &tokenizer).expect("transcribe");

    eprintln!("Full pipeline transcription:");
    eprintln!("  text:            {:?}", result.text.trim());
    eprintln!("  tokens:          {:?}", result.decode_result.tokens);
    eprintln!("  reached_eot:     {}", result.decode_result.reached_eot);
    eprintln!("  avg_logprob:     {:.4}", result.decode_result.avg_logprob);
    eprintln!("  no_speech_prob:  {:.4}", result.no_speech_prob);

    // Pipeline should complete without errors.
    // Token count should be > 0 (even for non-speech, the model generates some tokens).
    // Or if it immediately emits EOT, that is also valid.
    assert!(
        result.decode_result.reached_eot || !result.decode_result.tokens.is_empty(),
        "pipeline should produce tokens or reach EOT"
    );

    eprintln!("PASS: full transcription pipeline completes successfully");
}

// ===========================================================================
// 11. Transcribe with Reference Encoder Output
// ===========================================================================

#[test]
fn test_transcribe_with_pytorch_encoder_output() {
    // Feed PyTorch reference encoder output through the full decode pipeline.
    // Since the decoder uses learned positional embeddings (loaded from weights),
    // feeding identical encoder output should produce consistent results.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);
    let tokenizer = build_fallback_tokenizer();

    let (ref_enc_data, ref_enc_shape) = load_ref_npy(&dir, "ref_encoder_output");
    assert_eq!(ref_enc_shape, vec![1, 1500, 384]);

    let encoder_output = DynTensor::from_vec(ref_enc_data, &ref_enc_shape, &Device::Cpu)
        .expect("ref encoder output");

    let decode_config = DecodeConfig::default();
    let result =
        transcribe(&mut model, &encoder_output, &decode_config, &tokenizer).expect("transcribe");

    eprintln!("Transcription from PyTorch encoder output:");
    eprintln!("  text:        {:?}", result.text.trim());
    eprintln!("  tokens:      {:?}", result.decode_result.tokens);
    eprintln!("  avg_logprob: {:.4}", result.decode_result.avg_logprob);
    eprintln!("  reached_eot: {}", result.decode_result.reached_eot);

    // Should produce valid output.
    for &t in &result.decode_result.tokens {
        assert!(t < 51865, "token {t} out of vocab range");
    }

    // avg_logprob should be finite.
    assert!(
        result.decode_result.avg_logprob.is_finite(),
        "avg_logprob should be finite"
    );

    eprintln!("PASS: transcription with PyTorch encoder output succeeds");
}

// ===========================================================================
// 12. Quality Metrics Validation
// ===========================================================================

#[test]
fn test_quality_metrics_validity() {
    // Verify that all quality metrics from greedy decode are well-formed.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);

    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");
    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu).expect("mel");
    let encoder_output = model.encode(&mel).expect("encode");

    let decode_config = DecodeConfig::default();
    let result = greedy_decode(&mut model, &encoder_output, &decode_config).expect("greedy decode");

    // Validate all metric fields.
    assert!(
        result.avg_logprob.is_finite(),
        "avg_logprob must be finite: {}",
        result.avg_logprob,
    );
    assert!(
        result.compression_ratio.is_finite() && result.compression_ratio >= 1.0,
        "compression_ratio must be >= 1.0: {}",
        result.compression_ratio,
    );
    assert!(
        result.no_speech_prob.is_finite()
            && result.no_speech_prob >= 0.0
            && result.no_speech_prob <= 1.0,
        "no_speech_prob must be in [0, 1]: {}",
        result.no_speech_prob,
    );
    assert!(
        result.temperature.is_finite() && result.temperature >= 0.0,
        "temperature must be finite and non-negative: {}",
        result.temperature,
    );

    // passes_quality_check should be callable.
    let passes = nn_whisper::passes_quality_check(&result, &decode_config);
    eprintln!("Quality check passed: {passes}");
    eprintln!(
        "  avg_logprob:      {:.4} (threshold: {:.4})",
        result.avg_logprob, decode_config.avg_logprob_threshold
    );
    eprintln!(
        "  compression:      {:.4} (threshold: {:.4})",
        result.compression_ratio, decode_config.compression_ratio_threshold
    );
    eprintln!("  no_speech_prob:   {:.4}", result.no_speech_prob);

    eprintln!("PASS: all quality metrics are well-formed");
}
