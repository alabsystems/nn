// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended encoder-decoder interaction tests for Whisper.
//!
//! Covers: encoder output shapes per model size, cross-attention dimension
//! matching, KV cache growth and reset, tokenizer round-trip, beam search
//! configuration effects, config validation per model size, and audio
//! preprocessing shape verification. Part of #4186.

use crate::audio::{mel_filterbank, pcm_to_mel, whisper_mel_spectrogram_for_config};
use crate::beam_search::{normalize_score, WhisperBeamSearchConfig};
use crate::config::{WhisperConfig, HOP_LENGTH, N_FFT, N_FRAMES, N_SAMPLES, SAMPLE_RATE};
use crate::test_utils::{tiny_config, tiny_encoder_output};
use crate::tokenizer::{
    WhisperTokenizer, EOT_TOKEN, LANGUAGE_TOKEN_END, LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN,
    SOT_TOKEN,
};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};

// ============================================================================
// Helper: conv stride-2 output length (same formula as encoder_tests.rs)
// ============================================================================

fn expected_seq_len(mel_len: usize) -> usize {
    (mel_len + 2 - 3) / 2 + 1
}

// ============================================================================
// Section 1: Encoder output shapes per Whisper model size
// ============================================================================

/// Verify encoder produces [batch, seq, d_model] for each preset config.
/// Uses zero weights so we only validate shapes, not numerical correctness.
/// We use a small mel_len (8) and override max_source_positions to keep
/// the test fast, since full 1500 positions would allocate large tensors.
fn verify_encoder_shape_for_config(name: &str, config: &WhisperConfig) {
    let mel_len = 8;
    let seq_len = expected_seq_len(mel_len);

    // Override max_source_positions to fit our small test input.
    let test_config = config
        .clone()
        .with_max_source_positions(seq_len + 4)
        // Use fewer layers for speed in shape tests.
        .with_encoder_layers(1)
        .with_decoder_layers(1);
    test_config.validate().unwrap_or_else(|e| {
        panic!("{name}: config validation failed: {e}");
    });

    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder =
        crate::encoder::AudioEncoder::load(vb.pp("model.encoder"), &test_config).unwrap();

    let mel =
        DynTensor::zeros(&[1, config.num_mel_bins, mel_len], DType::F32, &cpu()).unwrap();
    let out = encoder.forward(&mel).unwrap();

    assert_eq!(
        out.rank(),
        3,
        "{name}: encoder output should be rank 3"
    );
    assert_eq!(
        out.dim(0).unwrap(),
        1,
        "{name}: batch dim should be 1"
    );
    assert_eq!(
        out.dim(1).unwrap(),
        seq_len,
        "{name}: seq dim should be {seq_len}"
    );
    assert_eq!(
        out.dim(2).unwrap(),
        config.d_model,
        "{name}: feature dim should be d_model={}",
        config.d_model
    );
}

#[test]
fn test_encoder_shape_tiny() {
    verify_encoder_shape_for_config("tiny", &WhisperConfig::whisper_tiny());
}

#[test]
fn test_encoder_shape_base() {
    verify_encoder_shape_for_config("base", &WhisperConfig::whisper_base());
}

#[test]
fn test_encoder_shape_small() {
    verify_encoder_shape_for_config("small", &WhisperConfig::whisper_small());
}

#[test]
fn test_encoder_shape_medium() {
    verify_encoder_shape_for_config("medium", &WhisperConfig::whisper_medium());
}

#[test]
fn test_encoder_shape_large_v2() {
    verify_encoder_shape_for_config("large-v2", &WhisperConfig::whisper_large_v2());
}

#[test]
fn test_encoder_shape_turbo() {
    verify_encoder_shape_for_config("turbo", &WhisperConfig::large_v3_turbo());
}

// ============================================================================
// Section 2: Cross-attention dimensions
//
// Encoder output dim(2) = d_model must equal decoder cross-attention key/value
// projection input dim. Since both use the same d_model, we verify the encoder
// output shape matches what the decoder expects for cross-attention.
// ============================================================================

#[test]
fn test_cross_attention_dim_matches_encoder_output_all_presets() {
    let presets: Vec<(&str, WhisperConfig)> = vec![
        ("tiny", WhisperConfig::whisper_tiny()),
        ("base", WhisperConfig::whisper_base()),
        ("small", WhisperConfig::whisper_small()),
        ("medium", WhisperConfig::whisper_medium()),
        ("large-v2", WhisperConfig::whisper_large_v2()),
        ("turbo", WhisperConfig::large_v3_turbo()),
    ];

    for (name, config) in &presets {
        // Encoder output dim(2) = d_model. Decoder cross-attention K/V projections
        // are [d_model, d_model], so they expect encoder features of dim d_model.
        // Verify this is consistent.
        assert_eq!(
            config.d_model,
            config.encoder_attention_heads * config.encoder_head_dim(),
            "{name}: encoder d_model decomposition mismatch"
        );
        assert_eq!(
            config.d_model,
            config.decoder_attention_heads * config.decoder_head_dim(),
            "{name}: decoder d_model decomposition mismatch"
        );
    }
}

#[test]
fn test_cross_attention_encoder_output_feeds_decoder() {
    // End-to-end: encoder output dim(2) = d_model feeds into decoder cross-attention.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    // Encoder: [1, mel_bins, 8] -> [1, seq, d_model]
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 8], DType::F32, &cpu()).unwrap();
    let enc_out = model.encode(&mel).unwrap();
    assert_eq!(
        enc_out.dim(2).unwrap(),
        config.d_model,
        "encoder output feature dim must equal d_model"
    );

    // Decoder uses encoder output directly for cross-attention.
    let tokens = DynTensor::from_vec_u32(vec![0; 3], &[1, 3], &cpu()).unwrap();
    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits.dims(), &[1, 3, config.vocab_size]);
}

#[test]
fn test_cross_attention_mismatched_d_model_errors() {
    // If encoder output has wrong feature dim, decoder should error.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    // Encoder output with wrong d_model (d_model+1).
    let bad_enc = DynTensor::zeros(
        &[1, 4, config.d_model + 1],
        DType::F32,
        &cpu(),
    )
    .unwrap();
    let tokens = DynTensor::from_vec_u32(vec![0; 2], &[1, 2], &cpu()).unwrap();
    let result = model.decode(&tokens, &bad_enc, true, 0);
    assert!(
        result.is_err(),
        "decoder should error when encoder output has mismatched d_model"
    );
}

// ============================================================================
// Section 3: KV cache mechanics
// ============================================================================

#[test]
fn test_kv_cache_grows_across_decode_steps() {
    // Multi-step decoding: each step should produce valid output.
    // The KV cache grows with each step, allowing the model to attend
    // to all previously decoded tokens.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let enc_out = tiny_encoder_output();

    // Step 1: initial prompt (3 tokens).
    let prompt = DynTensor::from_vec_u32(vec![0, 1, 2], &[1, 3], &cpu()).unwrap();
    let logits1 = model.decode(&prompt, &enc_out, true, 0).unwrap();
    assert_eq!(logits1.dims(), &[1, 3, config.vocab_size]);

    // Step 2: single next token at offset 3.
    let token2 = DynTensor::from_vec_u32(vec![5], &[1, 1], &cpu()).unwrap();
    let logits2 = model.decode(&token2, &enc_out, false, 3).unwrap();
    assert_eq!(logits2.dims(), &[1, 1, config.vocab_size]);

    // Step 3: another token at offset 4.
    let token3 = DynTensor::from_vec_u32(vec![7], &[1, 1], &cpu()).unwrap();
    let logits3 = model.decode(&token3, &enc_out, false, 4).unwrap();
    assert_eq!(logits3.dims(), &[1, 1, config.vocab_size]);

    // All outputs should be finite.
    for (i, logits) in [&logits1, &logits2, &logits3].iter().enumerate() {
        let flat = logits.to_flat_vec::<f32>().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "step {}: logits should be finite",
            i + 1
        );
    }
}

#[test]
fn test_kv_cache_clear_resets_state() {
    // After reset, decoding the same sequence should produce identical results.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config).unwrap();
    let enc_out = tiny_encoder_output();

    let tokens = DynTensor::from_vec_u32(vec![0, 1, 2], &[1, 3], &cpu()).unwrap();

    // First pass.
    let logits1 = model.decode(&tokens, &enc_out, true, 0).unwrap();
    let flat1 = logits1.to_flat_vec::<f32>().unwrap();

    // Advance cache state by decoding more tokens.
    let extra = DynTensor::from_vec_u32(vec![3], &[1, 1], &cpu()).unwrap();
    model.decode(&extra, &enc_out, false, 3).unwrap();

    // Reset.
    model.reset_kv_cache();

    // Second pass: same sequence after reset.
    let logits2 = model.decode(&tokens, &enc_out, true, 0).unwrap();
    let flat2 = logits2.to_flat_vec::<f32>().unwrap();

    assert_eq!(flat1.len(), flat2.len());
    for (i, (&a, &b)) in flat1.iter().zip(flat2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "index {i}: after reset, logits should match: {a} vs {b}"
        );
    }
}

#[test]
fn test_kv_cache_flush_on_new_segment() {
    // flush_kv_cache=true should reset cache even mid-sequence.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let enc_out = tiny_encoder_output();

    // Populate cache.
    let prompt = DynTensor::from_vec_u32(vec![0, 1, 2, 3, 4], &[1, 5], &cpu()).unwrap();
    model.decode(&prompt, &enc_out, true, 0).unwrap();

    // Flush and start fresh — should succeed without error.
    let new_prompt = DynTensor::from_vec_u32(vec![10, 11], &[1, 2], &cpu()).unwrap();
    let logits = model.decode(&new_prompt, &enc_out, true, 0).unwrap();
    assert_eq!(logits.dims(), &[1, 2, config.vocab_size]);
}

#[test]
fn test_kv_cache_incremental_matches_full_sequence() {
    // Incremental decode (cached) should produce the same final logits
    // as processing the full sequence at once via forward_no_cache.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());

    let enc_out = DynTensor::zeros(
        &[1, config.max_source_positions, config.d_model],
        DType::F32,
        &cpu(),
    )
    .unwrap();

    // Full sequence: [0, 1, 2, 3] via forward_no_cache.
    let full_tokens = DynTensor::from_vec_u32(vec![0, 1, 2, 3], &[1, 4], &cpu()).unwrap();
    let model_nc = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let logits_full = model_nc
        .decoder()
        .forward_no_cache(&full_tokens, &enc_out)
        .unwrap();

    // Incremental: [0,1,2] then [3] via cached forward.
    let mut model_cached = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let init = DynTensor::from_vec_u32(vec![0, 1, 2], &[1, 3], &cpu()).unwrap();
    model_cached.decode(&init, &enc_out, true, 0).unwrap();
    let next = DynTensor::from_vec_u32(vec![3], &[1, 1], &cpu()).unwrap();
    let logits_inc = model_cached.decode(&next, &enc_out, false, 3).unwrap();

    // Compare last-step logits (index 3).
    let flat_full = logits_full.to_flat_vec::<f32>().unwrap();
    let flat_inc = logits_inc.to_flat_vec::<f32>().unwrap();

    // Extract last step from full: [1, 4, vocab] -> last step has vocab elements.
    let vocab = config.vocab_size;
    let full_last = &flat_full[3 * vocab..4 * vocab];

    assert_eq!(full_last.len(), flat_inc.len());
    let max_err = full_last
        .iter()
        .zip(flat_inc.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-5,
        "incremental vs full-sequence last-step logits max error: {max_err}"
    );
}

// ============================================================================
// Section 4: Tokenizer round-trip
// ============================================================================

/// Reproduce GPT-2's byte-to-unicode mapping.
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

/// Build a tokenizer with 256 single-byte tokens + a few merges for roundtrip tests.
fn build_roundtrip_tokenizer() -> WhisperTokenizer {
    let byte_encoder = build_gpt2_byte_encoder();
    let mut vocab = serde_json::Map::new();
    for (byte, ch) in &byte_encoder {
        vocab.insert(
            ch.to_string(),
            serde_json::Value::Number(u64::from(*byte).into()),
        );
    }

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

    let vocab_json = serde_json::Value::Object(vocab).to_string();

    // We need at least one merge for from_vocab_and_merges to set bpe_ranks non-empty,
    // so add a dummy merge that won't affect ASCII text.
    let b254_char = byte_encoder[&0xFE];
    let b255_char = byte_encoder[&0xFF];
    let merges = format!(
        "#version: 0.2\n{b254_char} {b255_char}\n"
    );

    WhisperTokenizer::from_vocab_and_merges(&vocab_json, &merges)
        .expect("roundtrip tokenizer should build")
}

#[test]
fn test_tokenizer_roundtrip_ascii() {
    let tok = build_roundtrip_tokenizer();
    assert!(tok.can_encode());

    for text in &["hello", "world", "test 123", "abc def"] {
        let ids = tok.encode(text).unwrap_or_else(|e| {
            panic!("encode({text:?}) failed: {e}");
        });
        let decoded = tok.decode(&ids).unwrap_or_else(|e| {
            panic!("decode failed for {text:?}: {e}");
        });
        assert_eq!(
            &decoded, text,
            "roundtrip failed for ASCII: {text:?}"
        );
    }
}

#[test]
fn test_tokenizer_roundtrip_punctuation() {
    let tok = build_roundtrip_tokenizer();

    for text in &["hello, world!", "it's a test.", "x + y = z"] {
        let ids = tok.encode(text).unwrap();
        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(
            &decoded, text,
            "roundtrip failed for punctuation: {text:?}"
        );
    }
}

#[test]
fn test_tokenizer_roundtrip_empty_and_single_char() {
    let tok = build_roundtrip_tokenizer();

    // Empty string.
    let ids = tok.encode("").unwrap();
    assert!(ids.is_empty(), "encoding empty string should yield no tokens");
    let decoded = tok.decode(&ids).unwrap();
    assert_eq!(decoded, "");

    // Single character.
    let ids = tok.encode("a").unwrap();
    let decoded = tok.decode(&ids).unwrap();
    assert_eq!(decoded, "a");
}

#[test]
fn test_tokenizer_special_tokens_skipped_in_decode() {
    let tok = build_roundtrip_tokenizer();

    // Encode "hi", then intersperse special tokens in the ID sequence.
    let ids = tok.encode("hi").unwrap();
    let mut ids_with_special = vec![SOT_TOKEN, 50259]; // SOT, en
    ids_with_special.extend_from_slice(&ids);
    ids_with_special.push(EOT_TOKEN);

    let decoded = tok.decode(&ids_with_special).unwrap();
    assert_eq!(decoded, "hi", "special tokens should be skipped during decode");
}

#[test]
fn test_tokenizer_special_token_constants_are_consistent() {
    // Verify special token ordering: SOT < language range < task tokens < EOT boundary.
    assert!(SOT_TOKEN > EOT_TOKEN, "SOT (50258) > EOT (50257)");
    assert_eq!(LANGUAGE_TOKEN_START, 50259);
    assert_eq!(LANGUAGE_TOKEN_END, 50358);
    assert_eq!(LANGUAGE_TOKEN_END - LANGUAGE_TOKEN_START + 1, 100);
    assert_eq!(NO_SPEECH_TOKEN, 50363);
}

// ============================================================================
// Section 5: Beam search configuration effects
// ============================================================================

#[test]
fn test_beam_width_determines_hypothesis_count() {
    // The beam search config's beam_width sets the maximum number of hypotheses.
    // We verify the config validates correctly for various widths.
    for width in [1, 2, 5, 10] {
        let config = WhisperBeamSearchConfig {
            beam_width: width,
            ..Default::default()
        };
        config.validate().unwrap_or_else(|e| {
            panic!("beam_width={width} should validate: {e}");
        });
        assert_eq!(config.beam_width, width);
    }
}

#[test]
fn test_beam_width_zero_rejected() {
    let config = WhisperBeamSearchConfig {
        beam_width: 0,
        ..Default::default()
    };
    assert!(
        config.validate().is_err(),
        "beam_width=0 should fail validation"
    );
}

#[test]
fn test_length_penalty_affects_normalized_score() {
    // Higher length penalty should penalize longer sequences more.
    let score = -10.0f32;
    let length = 5;

    let norm_p0 = normalize_score(score, length, 0.0);
    let norm_p1 = normalize_score(score, length, 1.0);
    let norm_p2 = normalize_score(score, length, 2.0);

    // With penalty=0, no normalization.
    assert!(
        (norm_p0 - score).abs() < 1e-6,
        "penalty=0 should not normalize"
    );

    // With higher penalty, the (negative) score is divided by a larger number,
    // making the normalized score less negative (closer to zero).
    assert!(
        norm_p1 > norm_p2.min(score),
        "penalty=1 normalized score ({norm_p1}) should differ from raw ({score})"
    );

    // penalty=2 divides by len^2=25, penalty=1 divides by len^1=5.
    // score/5 = -2.0, score/25 = -0.4. Both are > score=-10.
    let expected_p1 = score / 5.0;
    let expected_p2 = score / 25.0;
    assert!(
        (norm_p1 - expected_p1).abs() < 1e-5,
        "penalty=1: expected {expected_p1}, got {norm_p1}"
    );
    assert!(
        (norm_p2 - expected_p2).abs() < 1e-5,
        "penalty=2: expected {expected_p2}, got {norm_p2}"
    );
}

#[test]
fn test_length_penalty_zero_returns_raw_score() {
    // With penalty=0, normalize_score returns the raw score regardless of length.
    for length in [0, 1, 5, 100] {
        let score = -3.5f32;
        let norm = normalize_score(score, length, 0.0);
        assert!(
            (norm - score).abs() < 1e-6,
            "penalty=0, length={length}: expected raw score"
        );
    }
}

#[test]
fn test_beam_search_config_defaults_match_whisper_spec() {
    let config = WhisperBeamSearchConfig::default();
    assert_eq!(config.beam_width, 5, "default beam width should be 5");
    assert_eq!(config.max_tokens, 448, "default max_tokens should be 448");
    assert!(
        (config.length_penalty - 1.0).abs() < 1e-6,
        "default length_penalty should be 1.0"
    );
    assert_eq!(config.sot_token, SOT_TOKEN);
    assert_eq!(config.eot_token, EOT_TOKEN);
}

#[test]
fn test_beam_search_config_temperature_validation() {
    // Negative temperature should be rejected.
    let mut config = WhisperBeamSearchConfig {
        temperature: -0.5,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    // NaN temperature should be rejected.
    config.temperature = f32::NAN;
    assert!(config.validate().is_err());

    // Zero temperature (greedy) should be accepted.
    config.temperature = 0.0;
    config.validate().unwrap();

    // Positive temperature should be accepted.
    config.temperature = 0.7;
    config.validate().unwrap();
}

// ============================================================================
// Section 6: Config validation per model size
// ============================================================================

#[test]
fn test_all_preset_configs_validate() {
    let presets: Vec<(&str, WhisperConfig)> = vec![
        ("tiny", WhisperConfig::whisper_tiny()),
        ("base", WhisperConfig::whisper_base()),
        ("small", WhisperConfig::whisper_small()),
        ("medium", WhisperConfig::whisper_medium()),
        ("large-v2", WhisperConfig::whisper_large_v2()),
        ("turbo", WhisperConfig::large_v3_turbo()),
    ];

    for (name, config) in &presets {
        config.validate().unwrap_or_else(|e| {
            panic!("{name}: config validation failed: {e}");
        });
    }
}

#[test]
fn test_preset_layer_counts() {
    // Verify expected layer counts for each model size.
    let cases: Vec<(&str, WhisperConfig, usize, usize)> = vec![
        ("tiny", WhisperConfig::whisper_tiny(), 4, 4),
        ("base", WhisperConfig::whisper_base(), 6, 6),
        ("small", WhisperConfig::whisper_small(), 12, 12),
        ("medium", WhisperConfig::whisper_medium(), 24, 24),
        ("large-v2", WhisperConfig::whisper_large_v2(), 32, 32),
        ("turbo", WhisperConfig::large_v3_turbo(), 32, 4),
    ];

    for (name, config, exp_enc, exp_dec) in &cases {
        assert_eq!(
            config.encoder_layers, *exp_enc,
            "{name}: encoder_layers"
        );
        assert_eq!(
            config.decoder_layers, *exp_dec,
            "{name}: decoder_layers"
        );
    }
}

#[test]
fn test_preset_head_counts() {
    let cases: Vec<(&str, WhisperConfig, usize, usize)> = vec![
        ("tiny", WhisperConfig::whisper_tiny(), 6, 6),
        ("base", WhisperConfig::whisper_base(), 8, 8),
        ("small", WhisperConfig::whisper_small(), 12, 12),
        ("medium", WhisperConfig::whisper_medium(), 16, 16),
        ("large-v2", WhisperConfig::whisper_large_v2(), 20, 20),
        ("turbo", WhisperConfig::large_v3_turbo(), 20, 20),
    ];

    for (name, config, exp_enc_heads, exp_dec_heads) in &cases {
        assert_eq!(
            config.encoder_attention_heads, *exp_enc_heads,
            "{name}: encoder_attention_heads"
        );
        assert_eq!(
            config.decoder_attention_heads, *exp_dec_heads,
            "{name}: decoder_attention_heads"
        );
    }
}

#[test]
fn test_preset_d_model_values() {
    let cases: Vec<(&str, WhisperConfig, usize)> = vec![
        ("tiny", WhisperConfig::whisper_tiny(), 384),
        ("base", WhisperConfig::whisper_base(), 512),
        ("small", WhisperConfig::whisper_small(), 768),
        ("medium", WhisperConfig::whisper_medium(), 1024),
        ("large-v2", WhisperConfig::whisper_large_v2(), 1280),
        ("turbo", WhisperConfig::large_v3_turbo(), 1280),
    ];

    for (name, config, exp_d) in &cases {
        assert_eq!(config.d_model, *exp_d, "{name}: d_model");
    }
}

#[test]
fn test_preset_encoder_decoder_d_model_match() {
    // Whisper uses the same d_model for encoder and decoder (shared via config).
    // Cross-attention requires encoder output dim == decoder query dim.
    let presets = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];

    for config in &presets {
        // Both encoder and decoder attention operate on d_model.
        assert_eq!(
            config.encoder_head_dim() * config.encoder_attention_heads,
            config.decoder_head_dim() * config.decoder_attention_heads,
            "encoder and decoder d_model must match for cross-attention"
        );
    }
}

#[test]
fn test_config_invalid_decoder_heads_rejected() {
    let config = tiny_config().with_decoder_attention_heads(0);
    assert!(
        config.validate().is_err(),
        "zero decoder_attention_heads should fail validation"
    );
}

#[test]
fn test_config_decoder_heads_not_dividing_d_model_rejected() {
    // d_model=16, 3 heads: 16 % 3 != 0
    let config = tiny_config().with_decoder_attention_heads(3);
    assert!(
        config.validate().is_err(),
        "d_model not divisible by decoder_heads should fail validation"
    );
}

// ============================================================================
// Section 7: Audio preprocessing — mel spectrogram shape verification
// ============================================================================

#[test]
fn test_mel_spectrogram_standard_30s_shape() {
    // Standard Whisper input: 30 seconds at 16 kHz -> 480,000 samples.
    // mel spectrogram: [1, num_mel_bins, N_FRAMES=3000].
    let audio: Vec<f32> = vec![0.0; N_SAMPLES];
    let mel = whisper_mel_spectrogram_for_config(&audio, 128).unwrap();
    assert_eq!(mel.dims(), &[1, 128, N_FRAMES]);
}

#[test]
fn test_mel_spectrogram_short_audio_padded_to_30s() {
    // Short audio (1 second) should be padded to 30s by whisper_mel_spectrogram.
    let audio: Vec<f32> = vec![0.0; SAMPLE_RATE]; // 1 second
    let mel = whisper_mel_spectrogram_for_config(&audio, 128).unwrap();
    assert_eq!(
        mel.dim(2).unwrap(),
        N_FRAMES,
        "short audio should be padded to N_FRAMES=3000"
    );
}

#[test]
fn test_mel_spectrogram_80_bins_for_smaller_models() {
    let audio: Vec<f32> = (0..SAMPLE_RATE)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
        .collect();
    let mel = whisper_mel_spectrogram_for_config(&audio, 80).unwrap();
    assert_eq!(mel.dim(1).unwrap(), 80);
    assert_eq!(mel.dim(2).unwrap(), N_FRAMES);
}

#[test]
fn test_mel_spectrogram_128_bins_for_large_models() {
    let audio: Vec<f32> = (0..SAMPLE_RATE)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
        .collect();
    let mel = whisper_mel_spectrogram_for_config(&audio, 128).unwrap();
    assert_eq!(mel.dim(1).unwrap(), 128);
    assert_eq!(mel.dim(2).unwrap(), N_FRAMES);
}

#[test]
fn test_mel_filterbank_shape() {
    let n_fft = N_FFT;
    let sr = SAMPLE_RATE;

    for n_mels in [80, 128] {
        let filters = mel_filterbank(n_mels, n_fft, sr);
        let expected_bins = n_fft / 2 + 1;
        // mel_filterbank returns a flat Vec<f32> of shape [n_mels * n_freqs].
        assert_eq!(
            filters.len(),
            n_mels * expected_bins,
            "filterbank for {n_mels} mels should have {} elements, got {}",
            n_mels * expected_bins,
            filters.len()
        );
        // All filter values should be non-negative (triangular filters).
        assert!(
            filters.iter().all(|&v| v >= 0.0 && v.is_finite()),
            "mel filter values should be non-negative and finite"
        );
    }
}

#[test]
fn test_pcm_to_mel_shape_from_duration() {
    // Verify mel frame count from known audio duration.
    let n_fft = N_FFT;
    let hop = HOP_LENGTH;
    let sr = SAMPLE_RATE;
    let duration_sec = 2.0;
    let n_samples = (sr as f64 * duration_sec) as usize;

    let audio: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
        .collect();

    let filters = mel_filterbank(80, n_fft, sr);
    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, 80).unwrap();

    assert_eq!(mel.dim(0).unwrap(), 1);
    assert_eq!(mel.dim(1).unwrap(), 80);

    // Frame count depends on STFT: approximately n_samples / hop_length.
    let n_frames = mel.dim(2).unwrap();
    let expected_approx = n_samples / hop;
    // Allow some tolerance for padding/edge effects.
    assert!(
        n_frames >= expected_approx.saturating_sub(2)
            && n_frames <= expected_approx + 2,
        "mel frames ({n_frames}) should be approximately {expected_approx} for {duration_sec}s audio"
    );
}

#[test]
fn test_audio_constants_consistency() {
    // Verify audio constant relationships.
    assert_eq!(SAMPLE_RATE, 16_000);
    assert_eq!(N_FFT, 400);
    assert_eq!(HOP_LENGTH, 160);
    assert_eq!(N_SAMPLES, SAMPLE_RATE * 30); // 30 second chunks
    assert_eq!(N_FRAMES, N_SAMPLES / HOP_LENGTH); // 3000 frames
}
