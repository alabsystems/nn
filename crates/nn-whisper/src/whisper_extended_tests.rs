// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Whisper model component tests covering tokenizer encode/decode
//! roundtrip, special token handling, model configuration presets,
//! mel spectrogram properties, and encoder/decoder architecture shapes
//! without real weights. Part of #4186.

use crate::audio::{mel_filterbank, pcm_to_mel, whisper_mel_spectrogram_for_config};
use crate::config::{WhisperConfig, HOP_LENGTH, N_FFT, N_FRAMES, SAMPLE_RATE};
use crate::test_utils::tiny_config;
use crate::tokenizer::{
    WhisperTokenizer, EOT_TOKEN, LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN, SOT_TOKEN,
};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};

// ============================================================================
// Helper: build a tokenizer with both vocab and merges for encode/decode tests
// ============================================================================

/// Minimal vocab + merges for BPE encode/decode roundtrip tests.
///
/// This vocab covers single-byte GPT-2 tokens so that any ASCII text can be
/// encoded as individual byte tokens even if no merge rule applies. The byte
/// encoder maps each ASCII byte to a unicode char; we include all 256 single-
/// byte tokens so the tokenizer can fall through to byte-level encoding.
fn build_test_tokenizer_with_merges() -> WhisperTokenizer {
    // Build vocab: all 256 single-byte tokens (GPT-2 byte-level BPE baseline).
    // The byte encoder maps bytes to unicode chars. We construct the forward
    // mapping and assign IDs 0..255 to each single-byte token string.
    let byte_encoder = build_gpt2_byte_encoder();
    let mut vocab = serde_json::Map::new();
    for (byte, ch) in &byte_encoder {
        let token_str = ch.to_string();
        vocab.insert(token_str, serde_json::Value::Number(u64::from(*byte).into()));
    }

    // Add a few multi-char tokens from common BPE merges.
    // "he" (merged from 'h' + 'e') at ID 256.
    let h_char = byte_encoder[&b'h'];
    let e_char = byte_encoder[&b'e'];
    let merged_he = format!("{h_char}{e_char}");
    vocab.insert(merged_he, serde_json::Value::Number(256.into()));

    // "ll" (merged from 'l' + 'l') at ID 257.
    let l_char = byte_encoder[&b'l'];
    let merged_ll = format!("{l_char}{l_char}");
    vocab.insert(merged_ll, serde_json::Value::Number(257.into()));

    // "lo" (merged from 'l' + 'o') at ID 258.
    let o_char = byte_encoder[&b'o'];
    let merged_lo = format!("{l_char}{o_char}");
    vocab.insert(merged_lo, serde_json::Value::Number(258.into()));

    // Add special tokens.
    vocab.insert(
        "<|endoftext|>".to_string(),
        serde_json::Value::Number(50257.into()),
    );
    vocab.insert(
        "<|startoftranscript|>".to_string(),
        serde_json::Value::Number(50258.into()),
    );
    vocab.insert(
        "<|en|>".to_string(),
        serde_json::Value::Number(50259.into()),
    );
    vocab.insert(
        "<|transcribe|>".to_string(),
        serde_json::Value::Number(50360.into()),
    );
    vocab.insert(
        "<|nospeech|>".to_string(),
        serde_json::Value::Number(50363.into()),
    );
    vocab.insert(
        "<|notimestamps|>".to_string(),
        serde_json::Value::Number(50364.into()),
    );

    let vocab_json = serde_json::Value::Object(vocab).to_string();

    // Merges: 'h' + 'e' -> 'he', 'l' + 'l' -> 'll', 'l' + 'o' -> 'lo'
    let h_str = h_char.to_string();
    let e_str = e_char.to_string();
    let l_str = l_char.to_string();
    let o_str = o_char.to_string();
    let merges = format!(
        "#version: 0.2\n{h_str} {e_str}\n{l_str} {l_str}\n{l_str} {o_str}\n"
    );

    WhisperTokenizer::from_vocab_and_merges(&vocab_json, &merges)
        .expect("test tokenizer should build")
}

/// Reproduce GPT-2's byte-to-unicode mapping (simplified).
fn build_gpt2_byte_encoder() -> std::collections::HashMap<u8, char> {
    let mut map = std::collections::HashMap::new();
    let mut n = 0u32;
    for b in 0u16..=255 {
        let byte = b as u8;
        let ch = match byte {
            b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF => u32::from(byte),
            _ => {
                let c = 256 + n;
                n += 1;
                c
            }
        };
        map.insert(byte, char::from_u32(ch).unwrap());
    }
    map
}

// ============================================================================
// Tokenizer: encode/decode roundtrip
// ============================================================================

#[test]
fn test_tokenizer_encode_decode_roundtrip() {
    let tok = build_test_tokenizer_with_merges();
    assert!(tok.can_encode(), "tokenizer should support encoding with merges");

    // Test with a simple ASCII string.
    // "hi" -> encodes via byte-level BPE -> decode should return "hi".
    let text = "hi";
    let ids = tok.encode(text).expect("encode should succeed");
    assert!(!ids.is_empty(), "encoded tokens should not be empty");
    let decoded = tok.decode(&ids).expect("decode should succeed");
    assert_eq!(
        decoded, text,
        "encode->decode roundtrip should return original text"
    );
}

#[test]
fn test_tokenizer_encode_decode_roundtrip_longer() {
    let tok = build_test_tokenizer_with_merges();

    // A longer string that exercises multiple BPE merges.
    let text = " hello";
    let ids = tok.encode(text).expect("encode should succeed");
    let decoded = tok.decode(&ids).expect("decode should succeed");
    assert_eq!(
        decoded, text,
        "encode->decode roundtrip for ' hello' should preserve text"
    );
}

#[test]
fn test_tokenizer_encode_decode_roundtrip_punctuation() {
    let tok = build_test_tokenizer_with_merges();

    let text = "a.b,c!";
    let ids = tok.encode(text).expect("encode should succeed");
    let decoded = tok.decode(&ids).expect("decode should succeed");
    assert_eq!(
        decoded, text,
        "encode->decode roundtrip should preserve punctuation"
    );
}

// ============================================================================
// Tokenizer: special tokens
// ============================================================================

#[test]
fn test_tokenizer_special_tokens() {
    let vocab = serde_json::json!({
        "hello": 0,
        "<|endoftext|>": 50257,
        "<|startoftranscript|>": 50258,
        "<|en|>": 50259,
        "<|transcribe|>": 50360,
        "<|nospeech|>": 50363,
        "<|notimestamps|>": 50364,
    });
    let tok = WhisperTokenizer::from_vocab_str(&vocab.to_string()).unwrap();

    // Verify special token IDs.
    assert_eq!(tok.token_id("<|endoftext|>"), Some(EOT_TOKEN));
    assert_eq!(tok.token_id("<|startoftranscript|>"), Some(SOT_TOKEN));
    assert_eq!(tok.token_id("<|en|>"), Some(LANGUAGE_TOKEN_START));
    assert_eq!(tok.token_id("<|nospeech|>"), Some(NO_SPEECH_TOKEN));

    // All special tokens should be classified as special.
    assert!(tok.is_special(EOT_TOKEN));
    assert!(tok.is_special(SOT_TOKEN));
    assert!(tok.is_special(LANGUAGE_TOKEN_START));
    assert!(tok.is_special(NO_SPEECH_TOKEN));

    // Decoding special tokens produces empty string (they are skipped).
    let text = tok
        .decode(&[SOT_TOKEN, LANGUAGE_TOKEN_START, 50360, 50364, EOT_TOKEN])
        .unwrap();
    assert_eq!(text, "", "decoding only special tokens should produce empty string");

    // Decoding regular + special tokens preserves only regular text.
    let text = tok.decode(&[SOT_TOKEN, 0, EOT_TOKEN]).unwrap();
    assert_eq!(text, "hello", "special tokens should be stripped in decode");
}

#[test]
fn test_tokenizer_language_token_lookup() {
    let vocab = serde_json::json!({
        "<|en|>": 50259,
        "<|fr|>": 50260,
        "<|de|>": 50261,
    });
    let tok = WhisperTokenizer::from_vocab_str(&vocab.to_string()).unwrap();

    assert_eq!(tok.language_token("en"), Some(50259));
    assert_eq!(tok.language_token("fr"), Some(50260));
    assert_eq!(tok.language_token("de"), Some(50261));
    assert_eq!(tok.language_token("ja"), None, "missing language should return None");
}

// ============================================================================
// Tokenizer: empty string
// ============================================================================

#[test]
fn test_tokenizer_empty_string() {
    let tok = build_test_tokenizer_with_merges();

    let ids = tok.encode("").expect("encode empty string should succeed");
    assert!(
        ids.is_empty(),
        "encoding empty string should produce empty token list, got {ids:?}"
    );

    let decoded = tok.decode(&ids).expect("decode empty tokens should succeed");
    assert_eq!(decoded, "", "decoding empty tokens should produce empty string");
}

// ============================================================================
// Model configuration: whisper-tiny
// ============================================================================

#[test]
fn test_whisper_config_tiny() {
    let c = WhisperConfig::whisper_tiny();
    assert_eq!(c.d_model, 384, "tiny d_model should be 384");
    assert_eq!(c.encoder_attention_heads, 6, "tiny should have 6 encoder heads");
    assert_eq!(c.decoder_attention_heads, 6, "tiny should have 6 decoder heads");
    assert_eq!(c.encoder_layers, 4, "tiny should have 4 encoder layers");
    assert_eq!(c.decoder_layers, 4, "tiny should have 4 decoder layers");
    assert_eq!(c.num_mel_bins, 80, "tiny uses 80 mel bins");
    assert_eq!(c.encoder_ffn_dim, 1536, "tiny ffn_dim = 4*384 = 1536");
    assert_eq!(c.decoder_ffn_dim, 1536);
    assert_eq!(c.vocab_size, 51865);
    assert_eq!(c.max_target_positions, 448);
    assert_eq!(c.max_source_positions, 1500);
    c.validate().expect("whisper-tiny config should be valid");
}

// ============================================================================
// Model configuration: whisper-base
// ============================================================================

#[test]
fn test_whisper_config_base() {
    let c = WhisperConfig::whisper_base();
    assert_eq!(c.d_model, 512, "base d_model should be 512");
    assert_eq!(c.encoder_attention_heads, 8, "base should have 8 encoder heads");
    assert_eq!(c.decoder_attention_heads, 8, "base should have 8 decoder heads");
    assert_eq!(c.encoder_layers, 6, "base should have 6 encoder layers");
    assert_eq!(c.decoder_layers, 6, "base should have 6 decoder layers");
    assert_eq!(c.num_mel_bins, 80, "base uses 80 mel bins");
    assert_eq!(c.encoder_ffn_dim, 2048, "base ffn_dim = 4*512 = 2048");
    assert_eq!(c.decoder_ffn_dim, 2048);
    assert_eq!(c.vocab_size, 51865);
    c.validate().expect("whisper-base config should be valid");
}

// ============================================================================
// Model configuration: attention heads divide hidden_size
// ============================================================================

#[test]
fn test_config_attention_heads_divide_hidden() {
    // Verify that hidden_size (d_model) is divisible by num_heads for all presets.
    let presets: Vec<(&str, WhisperConfig)> = vec![
        ("tiny", WhisperConfig::whisper_tiny()),
        ("base", WhisperConfig::whisper_base()),
        ("small", WhisperConfig::whisper_small()),
        ("medium", WhisperConfig::whisper_medium()),
        ("large-v2", WhisperConfig::whisper_large_v2()),
        ("large-v3-turbo", WhisperConfig::large_v3_turbo()),
    ];

    for (name, config) in &presets {
        assert_eq!(
            config.d_model % config.encoder_attention_heads,
            0,
            "{name}: d_model ({}) must be divisible by encoder_attention_heads ({})",
            config.d_model,
            config.encoder_attention_heads
        );
        assert_eq!(
            config.d_model % config.decoder_attention_heads,
            0,
            "{name}: d_model ({}) must be divisible by decoder_attention_heads ({})",
            config.d_model,
            config.decoder_attention_heads
        );
    }

    // Negative test: config where d_model is NOT divisible by heads.
    let bad_config = WhisperConfig::whisper_tiny().with_encoder_attention_heads(5);
    assert!(
        bad_config.validate().is_err(),
        "384 is not divisible by 5 -- validate should fail"
    );

    let bad_config2 = WhisperConfig::whisper_tiny().with_decoder_attention_heads(7);
    assert!(
        bad_config2.validate().is_err(),
        "384 is not divisible by 7 -- validate should fail"
    );
}

// ============================================================================
// Mel spectrogram: output shape [1, n_mels, time_frames]
// ============================================================================

#[test]
fn test_mel_spectrogram_output_shape() {
    // Standard Whisper mel spectrogram should have shape [1, num_mel_bins, N_FRAMES].
    let audio = vec![0.1f32; 16000]; // 1 second of audio
    let mel_128 = whisper_mel_spectrogram_for_config(&audio, 128).unwrap();
    assert_eq!(mel_128.rank(), 3);
    assert_eq!(mel_128.dim(0).unwrap(), 1, "batch dim should be 1");
    assert_eq!(mel_128.dim(1).unwrap(), 128, "mel bins should be 128");
    assert_eq!(
        mel_128.dim(2).unwrap(),
        N_FRAMES,
        "time frames should be {N_FRAMES} (30s padded)"
    );

    // 80 mel bins (for tiny/base/small/medium).
    let mel_80 = whisper_mel_spectrogram_for_config(&audio, 80).unwrap();
    assert_eq!(mel_80.dim(1).unwrap(), 80, "mel bins should be 80");
    assert_eq!(mel_80.dim(2).unwrap(), N_FRAMES);
}

// ============================================================================
// Mel spectrogram: silent input produces low energy
// ============================================================================

#[test]
fn test_mel_spectrogram_silent_input() {
    // Near-silent audio (very small amplitude) should produce low-energy mel values.
    // The log-mel normalization means silent audio produces values near the floor.
    let silent_audio = vec![1e-8_f32; 16000]; // near-zero amplitude
    let n_mels = 80;
    let filters = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);
    let mel = pcm_to_mel(&silent_audio, &filters, N_FFT, HOP_LENGTH, n_mels).unwrap();
    let flat = mel.to_flat_vec::<f32>().unwrap();

    // After log10 + clamp + affine normalization, near-silent audio should have
    // values clustered near the floor. Compute mean to verify low energy.
    let mean: f32 = flat.iter().sum::<f32>() / flat.len() as f32;

    // Normal speech typically produces values around 0.0..0.5 after normalization.
    // Silent audio should have a lower mean. The exact value depends on the
    // normalization, but it should be noticeably below typical speech levels.
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "all mel values should be finite even for silent input"
    );

    // Loud audio for comparison.
    let loud_audio: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
        .collect();
    let loud_mel = pcm_to_mel(&loud_audio, &filters, N_FFT, HOP_LENGTH, n_mels).unwrap();
    let loud_flat = loud_mel.to_flat_vec::<f32>().unwrap();
    let loud_mean: f32 = loud_flat.iter().sum::<f32>() / loud_flat.len() as f32;

    assert!(
        mean < loud_mean,
        "silent audio mean ({mean}) should be lower than loud audio mean ({loud_mean})"
    );
}

// ============================================================================
// Mel spectrogram: known frequency produces energy in expected band
// ============================================================================

#[test]
fn test_mel_spectrogram_known_frequency() {
    // A pure 1000 Hz tone should concentrate energy around the mel band
    // corresponding to 1000 Hz. With 80 mel bands spanning 0-8000 Hz
    // (Slaney scale), 1000 Hz is in the transition zone between linear
    // and log regions.
    let n_mels = 80;
    let filters = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);

    let freq = 1000.0_f32;
    let audio: Vec<f32> = (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin())
        .collect();

    let mel = pcm_to_mel(&audio, &filters, N_FFT, HOP_LENGTH, n_mels).unwrap();
    let n_frames = mel.dim(2).unwrap();
    let vals = mel.to_flat_vec::<f32>().unwrap();

    // Compute per-band mean energy.
    let mut band_means: Vec<f32> = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let sum: f32 = (0..n_frames).map(|t| vals[m * n_frames + t]).sum();
        band_means.push(sum / n_frames as f32);
    }

    let peak_band = band_means
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    // 1000 Hz is in the lower-to-mid range of mel bands. With 80 mel bands,
    // it should be roughly in the first third (below band ~30).
    assert!(
        peak_band < 40,
        "1000 Hz peak at mel band {peak_band}, expected in lower region (< 40 of {n_mels})"
    );

    // Also verify a high-frequency tone (6000 Hz) lands in upper bands.
    let high_freq = 6000.0_f32;
    let audio_high: Vec<f32> = (0..16000)
        .map(|i| {
            (2.0 * std::f32::consts::PI * high_freq * i as f32 / SAMPLE_RATE as f32).sin()
        })
        .collect();
    let mel_high = pcm_to_mel(&audio_high, &filters, N_FFT, HOP_LENGTH, n_mels).unwrap();
    let vals_high = mel_high.to_flat_vec::<f32>().unwrap();
    let n_frames_high = mel_high.dim(2).unwrap();

    let mut band_means_high: Vec<f32> = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let sum: f32 = (0..n_frames_high).map(|t| vals_high[m * n_frames_high + t]).sum();
        band_means_high.push(sum / n_frames_high as f32);
    }

    let peak_band_high = band_means_high
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    assert!(
        peak_band_high > peak_band,
        "6000 Hz peak band ({peak_band_high}) should be above 1000 Hz peak band ({peak_band})"
    );
}

// ============================================================================
// Encoder: output shape [batch, seq_len, d_model]
// ============================================================================

#[test]
fn test_encoder_output_shape() {
    // Verify encoder output has the expected [batch, seq_len, hidden_dim] shape.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    // Input: [1, num_mel_bins, mel_len]
    let mel_len = 16;
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, mel_len], DType::F32, &cpu()).unwrap();
    let out = model.encode(&mel).unwrap();

    assert_eq!(out.rank(), 3, "encoder output should be rank 3");
    assert_eq!(out.dim(0).unwrap(), 1, "batch dim");
    assert_eq!(
        out.dim(2).unwrap(),
        config.d_model,
        "last dim should be d_model"
    );

    // seq_len = (mel_len + 2*padding - kernel) / stride + 1
    // conv1: stride=1, pad=1, k=3 => out_len = mel_len
    // conv2: stride=2, pad=1, k=3 => out_len = (mel_len + 2 - 3)/2 + 1 = 8
    let expected_seq = (mel_len + 2 - 3) / 2 + 1;
    assert_eq!(
        out.dim(1).unwrap(),
        expected_seq,
        "seq_len should be {expected_seq} after conv stride-2"
    );

    // Verify output is finite.
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "all encoder output values should be finite"
    );
}

#[test]
fn test_encoder_output_shape_batch_2() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[2, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let out = model.encode(&mel).unwrap();

    assert_eq!(out.dim(0).unwrap(), 2, "batch size should be 2");
    assert_eq!(out.dim(2).unwrap(), config.d_model);
}

// ============================================================================
// Decoder: output shape [batch, seq_len, vocab_size]
// ============================================================================

#[test]
fn test_decoder_output_shape() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    // Encoder output: [1, audio_len, d_model]
    let enc_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();

    // Token IDs: [1, seq_len] as u32 tensor.
    let token_seq_len = 4;
    let tokens =
        DynTensor::from_vec_u32(vec![0; token_seq_len], &[1, token_seq_len], &cpu()).unwrap();

    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();

    assert_eq!(logits.rank(), 3, "decoder output should be rank 3");
    assert_eq!(logits.dim(0).unwrap(), 1, "batch dim");
    assert_eq!(
        logits.dim(1).unwrap(),
        token_seq_len,
        "seq_len should match input"
    );
    assert_eq!(
        logits.dim(2).unwrap(),
        config.vocab_size,
        "last dim should be vocab_size"
    );

    // Verify output is finite.
    let flat = logits.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "all decoder output values should be finite"
    );
}

#[test]
fn test_decoder_output_shape_single_token() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let enc_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    let tokens = DynTensor::from_vec_u32(vec![0], &[1, 1], &cpu()).unwrap();

    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits.dims(), &[1, 1, config.vocab_size]);
}

// ============================================================================
// Decoder: autoregressive multi-step decode shapes
// ============================================================================

#[test]
fn test_decoder_autoregressive_multi_step() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let enc_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();

    // Step 0: initial prompt of 3 tokens.
    let initial = DynTensor::from_vec_u32(vec![0, 1, 2], &[1, 3], &cpu()).unwrap();
    let logits0 = model.decode(&initial, &enc_out, true, 0).unwrap();
    assert_eq!(logits0.dims(), &[1, 3, config.vocab_size]);

    // Steps 1-3: single token each, incrementing position offset.
    for step in 0..3 {
        let t = DynTensor::from_vec_u32(vec![step as u32 + 3], &[1, 1], &cpu()).unwrap();
        let logits = model.decode(&t, &enc_out, false, 3 + step).unwrap();
        assert_eq!(
            logits.dims(),
            &[1, 1, config.vocab_size],
            "step {step} shape mismatch"
        );
    }
}

// ============================================================================
// Encoder: forward_no_cache produces valid output
// ============================================================================

#[test]
fn test_encoder_forward_no_cache_output() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 8], DType::F32, &cpu()).unwrap();
    let out = model.encoder().forward_no_cache(&mel).unwrap();

    assert_eq!(out.rank(), 3);
    assert_eq!(out.dim(0).unwrap(), 1);
    assert_eq!(out.dim(2).unwrap(), config.d_model);

    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));
}

// ============================================================================
// Decoder: forward_no_cache output shape matches vocab
// ============================================================================

#[test]
fn test_decoder_forward_no_cache_output() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let enc_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    let tokens = DynTensor::from_vec_u32(vec![0, 1, 2], &[1, 3], &cpu()).unwrap();

    let logits = model.decoder().forward_no_cache(&tokens, &enc_out).unwrap();
    assert_eq!(logits.rank(), 3);
    assert_eq!(logits.dim(0).unwrap(), 1);
    assert_eq!(logits.dim(1).unwrap(), 3);
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size);
}

// ============================================================================
// Tokenizer: decode_with_timestamps produces correct segments
// ============================================================================

#[test]
fn test_tokenizer_decode_with_timestamps_segments() {
    let vocab = serde_json::json!({
        "hello": 0,
        "\u{0120}world": 1,
        "<|endoftext|>": 50257,
        "<|startoftranscript|>": 50258,
    });
    let tok = WhisperTokenizer::from_vocab_str(&vocab.to_string()).unwrap();

    // Construct: <|0.00|> hello <|2.00|> <|2.00|> world <|4.00|>
    // Timestamps: 50365 = 0.00s, 50465 = 2.00s, 50565 = 4.00s
    let ts_0 = 50365; // 0.00s
    let ts_2 = 50365 + 100; // 2.00s
    let ts_4 = 50365 + 200; // 4.00s
    let tokens = vec![ts_0, 0, ts_2, ts_2, 1, ts_4];

    let segments = tok.decode_with_timestamps(&tokens).unwrap();
    assert_eq!(segments.len(), 2, "should produce 2 segments");

    assert_eq!(segments[0].text, "hello");
    assert!((segments[0].start.unwrap() - 0.0).abs() < 0.001);
    assert!((segments[0].end.unwrap() - 2.0).abs() < 0.001);

    assert_eq!(segments[1].text, " world");
    assert!((segments[1].start.unwrap() - 2.0).abs() < 0.001);
    assert!((segments[1].end.unwrap() - 4.0).abs() < 0.001);
}

// ============================================================================
// Tokenizer: timestamp value computation
// ============================================================================

#[test]
fn test_tokenizer_timestamp_values() {
    let tok = WhisperTokenizer::from_vocab_str("{}").unwrap();

    // Token 50365 = 0.00s (TIMESTAMP_BEGIN)
    assert_eq!(tok.timestamp_value(50365), Some(0.0));

    // Token 50366 = 0.02s
    assert!((tok.timestamp_value(50366).unwrap() - 0.02).abs() < 1e-10);

    // Token 50365 + 1500 = 51865 = 30.00s
    assert!((tok.timestamp_value(51865).unwrap() - 30.0).abs() < 1e-10);

    // Below TIMESTAMP_BEGIN is not a timestamp.
    assert_eq!(tok.timestamp_value(50364), None);
    assert_eq!(tok.timestamp_value(0), None);
}

// ============================================================================
// Config: validation catches invalid configurations
// ============================================================================

#[test]
fn test_config_validation_zero_fields() {
    // Each of these zero-field configs should fail validation.
    assert!(WhisperConfig::whisper_tiny().with_d_model(0).validate().is_err());
    assert!(
        WhisperConfig::whisper_tiny()
            .with_encoder_attention_heads(0)
            .validate()
            .is_err()
    );
    assert!(
        WhisperConfig::whisper_tiny()
            .with_decoder_attention_heads(0)
            .validate()
            .is_err()
    );
    assert!(WhisperConfig::whisper_tiny().with_vocab_size(0).validate().is_err());
    assert!(
        WhisperConfig::whisper_tiny()
            .with_num_mel_bins(0)
            .validate()
            .is_err()
    );
    assert!(
        WhisperConfig::whisper_tiny()
            .with_encoder_ffn_dim(0)
            .validate()
            .is_err()
    );
    assert!(
        WhisperConfig::whisper_tiny()
            .with_decoder_ffn_dim(0)
            .validate()
            .is_err()
    );
    assert!(
        WhisperConfig::whisper_tiny()
            .with_max_source_positions(0)
            .validate()
            .is_err()
    );
    assert!(
        WhisperConfig::whisper_tiny()
            .with_max_target_positions(0)
            .validate()
            .is_err()
    );
}

// ============================================================================
// Mel spectrogram: pcm_to_mel output values are finite and normalized
// ============================================================================

#[test]
fn test_mel_spectrogram_output_values_finite() {
    let n_mels = 80;
    let filters = mel_filterbank(n_mels, N_FFT, SAMPLE_RATE);

    // Various audio signals.
    let signals: Vec<(&str, Vec<f32>)> = vec![
        ("dc_offset", vec![0.5f32; 8000]),
        (
            "sine_440",
            (0..8000)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
                .collect(),
        ),
        ("quiet", vec![1e-6_f32; 8000]),
        ("loud", vec![0.99f32; 8000]),
    ];

    for (name, audio) in &signals {
        let mel = pcm_to_mel(audio, &filters, N_FFT, HOP_LENGTH, n_mels)
            .unwrap_or_else(|e| panic!("pcm_to_mel failed for {name}: {e}"));
        let flat = mel.to_flat_vec::<f32>().unwrap();
        for (i, &v) in flat.iter().enumerate() {
            assert!(
                v.is_finite(),
                "{name}: mel value at index {i} is not finite: {v}"
            );
        }
    }
}

// ============================================================================
// Full encode -> decode pipeline with zero weights
// ============================================================================

#[test]
fn test_encode_decode_pipeline_shape_consistency() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    // Encode mel input.
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let enc_out = model.encode(&mel).unwrap();

    assert_eq!(enc_out.rank(), 3);
    let audio_seq_len = enc_out.dim(1).unwrap();
    assert!(audio_seq_len > 0, "encoder should produce at least 1 frame");

    // Decode using encoder output.
    let tokens = DynTensor::from_vec_u32(vec![0, 1], &[1, 2], &cpu()).unwrap();
    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();

    assert_eq!(logits.dim(0).unwrap(), 1, "batch matches");
    assert_eq!(logits.dim(1).unwrap(), 2, "seq_len matches input tokens");
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size, "vocab_size matches config");
}
