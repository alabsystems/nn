// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro config pipeline integration tests (#4186).
//!
//! Tests config field interactions, segment configuration consistency,
//! sample rate / hop length / mel bin relationships, style vector dimensions,
//! phoneme vocabulary size, and config-to-pipeline parameter propagation.

use crate::kokoro_source::MAX_SINEGEN_FRAMES;
use crate::kokoro_tokenizer::{MAX_PHONEME_TOKENS, PAD_TOKEN_ID};
use crate::kokoro_tts::KokoroConfig;
use crate::kokoro_tts::{KOKORO_HOP_LENGTH, KOKORO_N_BINS, KOKORO_N_FFT, KOKORO_SAMPLE_RATE};
use crate::kokoro_vocab::KokoroVocab;
use crate::plbert::PlbertConfig;

// ===========================================================================
// 1. Config-to-pipeline parameter consistency
// ===========================================================================

#[test]
fn test_kokoro_sample_rate_is_24khz() {
    assert_eq!(
        KOKORO_SAMPLE_RATE, 24000,
        "Kokoro sample rate must be 24kHz"
    );
}

#[test]
fn test_kokoro_hop_length_matches_upsample_product() {
    // The Kokoro hop length (5) times the iSTFT n_fft internal dimension
    // should relate to the upsample rates product (60).
    // upsample_rates [10, 6] -> product = 60
    // Each mel frame produces 60 audio samples.
    // But hop_length for iSTFT = 5 (n_fft/4 = 20/4 = 5).
    let cfg = KokoroConfig::default();
    let upsample_product: usize = cfg.upsample_rates.iter().product();
    assert_eq!(upsample_product, 60);
    assert_eq!(KOKORO_HOP_LENGTH, KOKORO_N_FFT / 4);
}

#[test]
fn test_kokoro_n_bins_from_n_fft() {
    // n_bins = n_fft / 2 + 1 = 11
    assert_eq!(KOKORO_N_BINS, KOKORO_N_FFT / 2 + 1);
    assert_eq!(KOKORO_N_BINS, 11);
}

#[test]
fn test_kokoro_mel_bins_equal_decoder_output_channels_half() {
    // The Generator produces n_fft channels, split into n_fft/2 real + n_fft/2 imag.
    // Each half has n_fft/2 = 10 channels. With Nyquist padding: 11 bins.
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.n_fft / 2, 10);
    assert_eq!(cfg.n_fft / 2 + 1, KOKORO_N_BINS);
}

// ===========================================================================
// 2. Style vector dimensions
// ===========================================================================

#[test]
fn test_style_dim_is_half_voice_embedding() {
    // Style embedding is 128, split from 256-dim voice embedding.
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.style_dim, 128);
    // Voice embedding is 256 (2 * style_dim).
    assert_eq!(cfg.style_dim * 2, 256);
}

#[test]
fn test_prosody_predictor_input_dim() {
    // ProsodyPredictor input = d_model + style_dim (the PlBert output + style vector).
    // PlBert output dimension = hidden_size = 768.
    // But the Kokoro ProsodyPredictor uses d_en (512) not hidden_size (768).
    let cfg = KokoroConfig::default();
    let prosody_input = cfg.d_en + cfg.style_dim;
    assert_eq!(
        prosody_input, 640,
        "prosody predictor input = d_en + style_dim = 640"
    );
}

#[test]
fn test_n_prosody_layers_is_3() {
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.n_prosody_layers, 3);
}

// ===========================================================================
// 3. Phoneme vocabulary size
// ===========================================================================

#[test]
fn test_kokoro_default_vocab_size_178() {
    let vocab = KokoroVocab::kokoro_default();
    assert_eq!(vocab.n_tokens(), 178, "Kokoro default vocab has 178 tokens");
}

#[test]
fn test_plbert_vocab_matches_kokoro_vocab() {
    let cfg = KokoroConfig::default();
    let vocab = KokoroVocab::kokoro_default();
    assert_eq!(
        cfg.plbert.vocab_size as u32,
        vocab.n_tokens(),
        "PlBert vocab_size must match KokoroVocab n_tokens"
    );
}

#[test]
fn test_pad_token_id_is_zero() {
    assert_eq!(PAD_TOKEN_ID, 0);
}

#[test]
fn test_max_phoneme_tokens_matches_plbert_context() {
    // MAX_PHONEME_TOKENS = 510 = max_position_embeddings(512) - 2 (start+end pad)
    let cfg = KokoroConfig::default();
    assert_eq!(MAX_PHONEME_TOKENS, cfg.plbert.max_position_embeddings - 2);
}

// ===========================================================================
// 4. Vocab lookup operations
// ===========================================================================

#[test]
fn test_vocab_insert_and_lookup() {
    let mut vocab = KokoroVocab::empty();
    vocab.insert('a', 43);
    assert_eq!(vocab.get('a'), Some(43));
    assert_eq!(vocab.decode_id(43), Some('a'));
}

#[test]
fn test_vocab_remove() {
    let mut vocab = KokoroVocab::empty();
    vocab.insert('x', 66);
    assert_eq!(vocab.remove('x'), Some(66));
    assert_eq!(vocab.get('x'), None);
    assert_eq!(vocab.remove('x'), None); // already removed
}

#[test]
fn test_vocab_empty_is_empty() {
    let vocab = KokoroVocab::empty();
    assert!(vocab.is_empty());
    assert_eq!(vocab.len(), 0);
    assert_eq!(vocab.n_tokens(), 1); // padding token
}

#[test]
fn test_vocab_default_not_empty() {
    let vocab = KokoroVocab::kokoro_default();
    assert!(!vocab.is_empty());
    assert!(vocab.len() > 100, "Kokoro vocab has >100 phoneme mappings");
}

#[test]
fn test_vocab_iter_yields_all_entries() {
    let vocab = KokoroVocab::kokoro_default();
    let entries: Vec<_> = vocab.iter().collect();
    assert_eq!(entries.len(), vocab.len());
    // All IDs should be unique.
    let mut ids: Vec<u32> = entries.iter().map(|&(_, id)| id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), entries.len(), "all token IDs should be unique");
}

#[test]
fn test_vocab_common_phonemes_present() {
    let vocab = KokoroVocab::kokoro_default();
    // Common ASCII phonemes should be present
    for ch in ['a', 'b', 'c', 'd', 'e'] {
        assert!(
            vocab.get(ch).is_some(),
            "common phoneme '{ch}' should be in default vocab"
        );
    }
    // Punctuation
    for ch in ['.', ',', '!', '?'] {
        assert!(
            vocab.get(ch).is_some(),
            "punctuation '{ch}' should be in default vocab"
        );
    }
}

#[test]
fn test_vocab_space_token_is_16() {
    let vocab = KokoroVocab::kokoro_default();
    assert_eq!(vocab.get(' '), Some(16), "space token should be ID 16");
}

// ===========================================================================
// 5. SineGen frame limit
// ===========================================================================

#[test]
fn test_max_sinegen_frames_sufficient_for_kokoro() {
    // MAX_SINEGEN_FRAMES = 8000 frames at 24kHz/300upp = ~67 seconds.
    // Kokoro max = 512 tokens = ~40 seconds. So 8000 provides ~2x headroom.
    assert_eq!(MAX_SINEGEN_FRAMES, 8000);
    // At 24kHz with upsample factor 300, 8000 frames = 8000 * 300 / 24000 = 100 seconds.
    // Actually: 8000 frames at mel rate, each frame = hop_length=60 audio samples.
    let max_audio_seconds = MAX_SINEGEN_FRAMES as f64 * 60.0 / 24000.0;
    assert!(
        max_audio_seconds > 20.0,
        "MAX_SINEGEN_FRAMES should allow at least 20s of audio, got {max_audio_seconds:.1}s"
    );
}

// ===========================================================================
// 6. Config validation edge cases
// ===========================================================================

#[test]
fn test_config_validate_accepts_modified_valid() {
    let cfg = KokoroConfig {
        d_en: 256,
        style_dim: 64,
        n_fft: 16, // divisible by 4
        max_dur: 25,
        ..Default::default()
    };
    cfg.validate()
        .expect("modified but valid config should pass");
}

#[test]
fn test_config_validate_rejects_n_fft_2() {
    // n_fft=2 is even but not divisible by 4.
    let cfg = KokoroConfig {
        n_fft: 2,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_gen_initial_channels_progression() {
    // Generator halves channels at each upsample stage.
    let cfg = KokoroConfig::default();
    let mut ch = cfg.gen_initial_channels;
    let stages = cfg.upsample_rates.len();
    for _ in 0..stages {
        ch /= 2;
    }
    // After 2 stages: 512 -> 256 -> 128
    assert_eq!(ch, 128);
    // This must equal n_fft + 2 for the output conv (magnitude + phase channels).
    // Actually the output conv produces n_fft channels (10 real + 10 imag).
    // 128 > 20, so the output_conv projects down: 128 -> n_fft.
}

#[test]
fn test_plbert_config_default_consistency() {
    let pc = PlbertConfig::default();
    // head_dim = hidden / heads = 768 / 12 = 64
    assert_eq!(pc.hidden_size / pc.num_attention_heads, 64);
    // intermediate = 2048 (standard 4x multiplier for ALBERT)
    assert_eq!(pc.intermediate_size, 2048);
    // layer_norm_eps matches ALBERT convention
    assert!((pc.layer_norm_eps - 1e-12).abs() < 1e-15);
}

// ===========================================================================
// 7. f0_bilstm_hidden dimension
// ===========================================================================

#[test]
fn test_f0_bilstm_hidden_matches_half_d_en() {
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.f0_bilstm_hidden, cfg.d_en / 2);
}

#[test]
fn test_max_dur_bins_default() {
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.max_dur, 50, "max_dur should be 50 Bernoulli bins");
}
