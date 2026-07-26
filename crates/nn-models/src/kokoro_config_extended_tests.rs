// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kokoro configuration tests (#4186).
//!
//! Tests model configuration defaults, style/speaker config, signal processing
//! config relationships, tokenizer/vocab properties, and cross-config consistency
//! that go beyond the basic tests in `kokoro_config_tests.rs` and
//! `kokoro_config_pipeline_tests.rs`.

use crate::istft::IstftParams;
use crate::kokoro_forward_stft::KokoroForwardStft;
use crate::kokoro_tokenizer::{KokoroTokenizer, KokoroVocab, MAX_PHONEME_TOKENS, PAD_TOKEN_ID};
use crate::kokoro_tts::{
    KokoroConfig, KOKORO_HOP_LENGTH, KOKORO_N_BINS, KOKORO_N_FFT, KOKORO_SAMPLE_RATE,
};
use crate::plbert::PlbertConfig;
use crate::stft::StftParams;

// ===========================================================================
// 1. Model Configuration — sample rate, dimensions, vocab, seq length
// ===========================================================================

#[test]
fn test_kokoro_default_sample_rate() {
    assert_eq!(
        KOKORO_SAMPLE_RATE, 24000,
        "Kokoro sample rate must be 24000 Hz"
    );
}

#[test]
fn test_kokoro_model_dimensions() {
    let cfg = KokoroConfig::default();
    // Encoder dimension (d_en) = 512
    assert_eq!(cfg.d_en, 512, "encoder dimension d_en should be 512");
    // Number of prosody predictor layers = 3
    assert_eq!(cfg.n_prosody_layers, 3, "n_prosody_layers should be 3");
    // PlBert hidden size = 768 (ALBERT-style)
    assert_eq!(
        cfg.plbert.hidden_size, 768,
        "PlBert hidden_size should be 768"
    );
    // PlBert num layers (shared) = 12
    assert_eq!(
        cfg.plbert.num_hidden_layers, 12,
        "PlBert num_hidden_layers should be 12"
    );
    // Generator initial channels = 512
    assert_eq!(
        cfg.gen_initial_channels, 512,
        "gen_initial_channels should be 512"
    );
    // F0 BiLSTM hidden = 256
    assert_eq!(cfg.f0_bilstm_hidden, 256, "f0_bilstm_hidden should be 256");
}

#[test]
fn test_kokoro_vocab_size() {
    // Kokoro-82M uses 178 phoneme tokens (IDs 0-177, with gaps)
    let vocab = KokoroVocab::kokoro_default();
    assert_eq!(
        vocab.n_tokens(),
        178,
        "Kokoro default vocab should have 178 tokens"
    );
    // PlBert embedding table should match
    let cfg = KokoroConfig::default();
    assert_eq!(
        cfg.plbert.vocab_size, 178,
        "PlBert vocab_size should match Kokoro vocab n_tokens"
    );
}

#[test]
fn test_kokoro_max_seq_length() {
    // PlBert max_position_embeddings = 512
    let cfg = KokoroConfig::default();
    assert_eq!(
        cfg.plbert.max_position_embeddings, 512,
        "PlBert max position embeddings should be 512"
    );
    // Maximum phoneme tokens = 512 - 2 (start + end padding) = 510
    assert_eq!(
        MAX_PHONEME_TOKENS,
        cfg.plbert.max_position_embeddings - 2,
        "MAX_PHONEME_TOKENS should be max_position_embeddings - 2"
    );
}

// ===========================================================================
// 2. Style/Speaker Configuration
// ===========================================================================

#[test]
fn test_kokoro_style_embedding_dim() {
    let cfg = KokoroConfig::default();
    assert_eq!(
        cfg.style_dim, 128,
        "style embedding dimension should be 128"
    );
    // Voice embedding is 256 = 2 * style_dim (split into decoder + prosody halves)
    assert_eq!(
        cfg.style_dim * 2,
        256,
        "full voice embedding should be 2 * style_dim = 256"
    );
}

#[test]
fn test_kokoro_num_styles_via_config() {
    // Kokoro does not define a fixed number of styles in the config;
    // styles are loaded from voice packs (safetensors). Verify the style_dim
    // is consistent with the prosody predictor input.
    let cfg = KokoroConfig::default();
    // ProsodyPredictor input = d_en + style_dim = 512 + 128 = 640
    let prosody_input_dim = cfg.d_en + cfg.style_dim;
    assert_eq!(
        prosody_input_dim, 640,
        "prosody predictor input dimension should be d_en + style_dim = 640"
    );
}

// ===========================================================================
// 3. Signal Processing Config
// ===========================================================================

#[test]
fn test_stft_config_defaults() {
    // Silero VAD STFT defaults
    let params = StftParams::default();
    assert_eq!(params.n_fft, 256, "StftParams default n_fft should be 256");
    assert_eq!(
        params.hop_length, 128,
        "StftParams default hop_length should be 128"
    );
    assert_eq!(
        params.n_freqs, 129,
        "StftParams default n_freqs should be 129"
    );
    assert_eq!(
        params.pad_right, 64,
        "StftParams default pad_right should be 64"
    );
}

#[test]
fn test_stft_frequency_bins() {
    // General relationship: n_freqs = n_fft / 2 + 1
    let params = StftParams::new(256, 128);
    assert_eq!(
        params.n_freqs,
        params.n_fft / 2 + 1,
        "frequency bins should be n_fft/2 + 1"
    );

    // Kokoro-specific: n_fft=20 -> 11 bins
    assert_eq!(
        KOKORO_N_BINS,
        KOKORO_N_FFT / 2 + 1,
        "Kokoro n_bins should be n_fft/2 + 1"
    );
    assert_eq!(KOKORO_N_BINS, 11, "Kokoro should have 11 frequency bins");
}

#[test]
fn test_kokoro_forward_stft_config() {
    // KokoroForwardStft uses n_fft=20, hop_length=5
    let stft = KokoroForwardStft::new(KOKORO_N_FFT, KOKORO_HOP_LENGTH, &nn_core::Device::Cpu)
        .expect("forward STFT creation should succeed");
    assert_eq!(
        stft.n_bins(),
        KOKORO_N_BINS,
        "forward STFT n_bins should match KOKORO_N_BINS"
    );
}

#[test]
fn test_kokoro_istft_params() {
    // Kokoro iSTFT: n_fft=20, hop_length=5, not normalized, no center trim
    let params = IstftParams::new(KOKORO_N_FFT, KOKORO_HOP_LENGTH, false, false)
        .expect("Kokoro iSTFT params should be valid");
    assert_eq!(params.n_fft, 20);
    assert_eq!(params.hop_length, 5);
    assert!(!params.normalized);
    assert!(!params.center);
}

#[test]
fn test_kokoro_hop_length_is_n_fft_over_4() {
    // Kokoro hop_length = n_fft / 4 = 20 / 4 = 5
    assert_eq!(
        KOKORO_HOP_LENGTH,
        KOKORO_N_FFT / 4,
        "Kokoro hop_length should be n_fft / 4"
    );
}

#[test]
fn test_mel_config_upsample_product() {
    // The upsample product (10 * 6 = 60) defines the mel-to-audio stride.
    // Each mel frame produces 60 audio samples.
    let cfg = KokoroConfig::default();
    let mel_stride: usize = cfg.upsample_rates.iter().product();
    assert_eq!(mel_stride, 60, "mel-to-audio stride should be 60");

    // Audio samples per second = KOKORO_SAMPLE_RATE = 24000
    // Mel frames per second = 24000 / 60 = 400
    let mel_fps = KOKORO_SAMPLE_RATE / mel_stride;
    assert_eq!(mel_fps, 400, "mel frame rate should be 400 fps at 24kHz");
}

// ===========================================================================
// 4. Tokenizer
// ===========================================================================

#[test]
fn test_tokenizer_phoneme_set() {
    let vocab = KokoroVocab::kokoro_default();
    assert!(
        !vocab.is_empty(),
        "default Kokoro phoneme set must be non-empty"
    );
    // Should have at least 100 phoneme mappings (actual: ~177 chars mapped)
    assert!(
        vocab.len() > 100,
        "Kokoro vocab should have >100 phoneme mappings, got {}",
        vocab.len()
    );
}

#[test]
fn test_tokenizer_special_tokens() {
    // Pad token is ID 0
    assert_eq!(PAD_TOKEN_ID, 0, "pad token ID should be 0");

    // Verify encoding wraps with pad tokens
    let tokenizer = KokoroTokenizer::kokoro_default();
    let encoded = tokenizer.encode("a").expect("encoding 'a' should succeed");
    // Result should be [0, token_id_for_a, 0]
    assert_eq!(encoded[0], PAD_TOKEN_ID, "first token should be PAD");
    assert_eq!(
        *encoded.last().unwrap(),
        PAD_TOKEN_ID,
        "last token should be PAD"
    );
    assert_eq!(
        encoded.len(),
        3,
        "single char encode should be [PAD, id, PAD]"
    );
}

#[test]
fn test_tokenizer_max_length() {
    let tokenizer = KokoroTokenizer::kokoro_default();
    assert_eq!(
        tokenizer.max_tokens(),
        MAX_PHONEME_TOKENS,
        "tokenizer max_tokens should be MAX_PHONEME_TOKENS"
    );
    assert_eq!(
        tokenizer.max_tokens(),
        510,
        "max phoneme tokens should be 510"
    );
}

#[test]
fn test_tokenizer_unknown_chars_dropped() {
    // Characters not in the vocabulary are silently dropped
    let tokenizer = KokoroTokenizer::kokoro_default();
    // Unicode snowman is not in Kokoro vocab
    let encoded = tokenizer
        .encode("\u{2603}")
        .expect("encoding unknown char should succeed");
    // Result should be just [PAD, PAD] since the char is dropped
    assert_eq!(encoded, vec![PAD_TOKEN_ID, PAD_TOKEN_ID]);
}

#[test]
fn test_tokenizer_empty_input() {
    let tokenizer = KokoroTokenizer::kokoro_default();
    let encoded = tokenizer
        .encode("")
        .expect("encoding empty string should succeed");
    assert_eq!(encoded, vec![PAD_TOKEN_ID, PAD_TOKEN_ID]);
}

#[test]
fn test_tokenizer_with_validated_vocab() {
    let vocab = KokoroVocab::kokoro_default();
    // Validate against embedding size 178 (matching PlBert vocab_size)
    let tokenizer = KokoroTokenizer::with_validated_vocab(vocab.clone(), 178)
        .expect("validation against size 178 should succeed");
    assert_eq!(tokenizer.vocab().n_tokens(), 178);

    // Validation should fail if embedding size is too small
    let result = KokoroTokenizer::with_validated_vocab(vocab, 100);
    assert!(
        result.is_err(),
        "validation against size 100 should fail (max token ID is 177)"
    );
}

#[test]
fn test_vocab_roundtrip_insert_decode() {
    let mut vocab = KokoroVocab::empty();
    let id = vocab.insert_auto('X');
    assert_eq!(id, 1, "first auto-insert should get ID 1 (after padding=0)");
    assert_eq!(vocab.get('X'), Some(1));
    assert_eq!(vocab.decode_id(1), Some('X'));
    assert_eq!(vocab.n_tokens(), 2); // padding + X
}

// ===========================================================================
// 5. Cross-config consistency
// ===========================================================================

#[test]
fn test_plbert_embedding_dim_matches_vocab_style() {
    let cfg = KokoroConfig::default();
    let pc = &cfg.plbert;
    // PlBert factorized embedding: 128-dim -> 768-dim via projection
    assert_eq!(pc.embedding_dim, 128);
    assert_eq!(pc.hidden_size, 768);
    assert!(
        pc.embedding_dim < pc.hidden_size,
        "ALBERT factorized: embedding_dim < hidden_size"
    );
}

#[test]
fn test_config_n_fft_and_istft_consistency() {
    let cfg = KokoroConfig::default();
    // Config n_fft must be divisible by 4 (validation rule)
    assert!(cfg.n_fft.is_multiple_of(4));
    // Config n_fft matches the KOKORO_N_FFT constant
    assert_eq!(cfg.n_fft, KOKORO_N_FFT);
    // iSTFT params can be created from config n_fft
    let params = IstftParams::new(cfg.n_fft, cfg.n_fft / 4, false, false)
        .expect("iSTFT params from config should be valid");
    assert_eq!(params.n_fft, cfg.n_fft);
}

#[test]
fn test_forward_stft_n_fft_even_required() {
    // Forward STFT requires n_fft to be even
    let result = KokoroForwardStft::new(21, 5, &nn_core::Device::Cpu);
    assert!(result.is_err(), "odd n_fft should be rejected");

    let result = KokoroForwardStft::new(0, 5, &nn_core::Device::Cpu);
    assert!(result.is_err(), "zero n_fft should be rejected");
}

#[test]
fn test_forward_stft_hop_length_nonzero_required() {
    let result = KokoroForwardStft::new(20, 0, &nn_core::Device::Cpu);
    assert!(result.is_err(), "zero hop_length should be rejected");
}

#[test]
fn test_stft_params_custom() {
    // Verify StftParams::new() computes derived fields correctly
    let params = StftParams::new(512, 256);
    assert_eq!(params.n_fft, 512);
    assert_eq!(params.hop_length, 256);
    assert_eq!(params.n_freqs, 257, "n_freqs = 512/2 + 1 = 257");
    assert_eq!(params.pad_right, 128, "pad_right = 512/4 = 128");
}

#[test]
fn test_plbert_config_defaults_match_kokoro() {
    // PlbertConfig::default() should produce the same values as KokoroConfig::default().plbert
    let standalone = PlbertConfig::default();
    let from_kokoro = KokoroConfig::default().plbert;
    assert_eq!(standalone.vocab_size, from_kokoro.vocab_size);
    assert_eq!(standalone.embedding_dim, from_kokoro.embedding_dim);
    assert_eq!(standalone.hidden_size, from_kokoro.hidden_size);
    assert_eq!(
        standalone.num_attention_heads,
        from_kokoro.num_attention_heads
    );
    assert_eq!(standalone.intermediate_size, from_kokoro.intermediate_size);
    assert_eq!(
        standalone.max_position_embeddings,
        from_kokoro.max_position_embeddings
    );
    assert_eq!(standalone.num_hidden_layers, from_kokoro.num_hidden_layers);
    assert!((standalone.layer_norm_eps - from_kokoro.layer_norm_eps).abs() < 1e-15);
}
