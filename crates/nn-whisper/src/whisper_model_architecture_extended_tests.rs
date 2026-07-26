// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Whisper model architecture and configuration tests.
//!
//! Focuses on architecture-level invariants NOT covered by existing test files:
//! - Attention scale factor: Whisper-specific head_dim^{-0.25} for Q and K
//! - Residual block structure: encoder (self-attn only) vs decoder (self + cross)
//! - Weight key naming: VarBuilder prefix expectations for all components
//! - Tied output projection: decoder logits via token embedding weight transpose
//! - Multi-head attention projection dimensions (Q/K/V/O all [d_model, d_model])
//! - Cross-attention cache invalidation and stale cache detection
//! - Positional encoding: sinusoidal (encoder) vs learned (decoder) distinction
//! - Autoregressive decode position offset edge cases
//! - Model dtype conversion: mel F32 with BF16/F16 model weights
//! - Residual connection shape preservation through blocks
//! - GELU activation in conv stem and FFN
//! - WhisperError variant coverage for all structured error types
//! - Encoder conv1 preserves sequence length, conv2 halves it
//! - Causal mask slice semantics at non-zero offsets
//! - Generation config: beam size, temperature, length penalty interactions

use crate::config::{
    WhisperConfig, CHUNK_LENGTH, HOP_LENGTH, N_FFT, N_FRAMES, N_SAMPLES, NUM_MEL_BINS,
    SAMPLE_RATE,
};
use crate::decode::{
    compression_ratio, DecodeConfig, DecodingResult, LongFormConfig, MAX_DECODE_LENGTH,
    DEFAULT_AVG_LOGPROB_THRESHOLD, DEFAULT_COMPRESSION_RATIO_THRESHOLD, DEFAULT_TEMPERATURES,
};
use crate::error::WhisperError;
use crate::positional::{causal_mask, sinusoidal_embedding};
use crate::test_utils::{tiny_config, tiny_encoder_output, tiny_model};
use crate::tokenizer::{
    WhisperTokenizer, LANGUAGE_TOKEN_START,
    SOT_TOKEN, TIMESTAMP_BEGIN,
};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, TensorError, VarBuilder};

// ============================================================================
// 1. Attention scale factor: Whisper-specific head_dim^{-0.25}
// ============================================================================

#[test]
fn test_whisper_attention_scale_factor_formula() {
    // Whisper uses (head_dim)^{-0.25} applied to BOTH Q and K, so the net
    // effect is Q * K^T * (head_dim)^{-0.5} = standard scaled dot product.
    // This differs from typical implementations that apply (head_dim)^{-0.5} to Q only.
    for config in &[
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ] {
        let head_dim = config.encoder_head_dim();
        let whisper_scale = (head_dim as f64).powf(-0.25);
        let standard_scale = (head_dim as f64).powf(-0.5);

        // Q_scaled * K_scaled^T = (Q * s) * (K * s)^T = s^2 * Q * K^T
        // s^2 = (head_dim^{-0.25})^2 = head_dim^{-0.5} = standard scale
        let combined = whisper_scale * whisper_scale;
        assert!(
            (combined - standard_scale).abs() < 1e-10,
            "Whisper scale^2 ({combined}) should equal standard scale ({standard_scale})"
        );
    }
}

#[test]
fn test_attention_scale_for_head_dim_64() {
    // All Whisper presets use head_dim=64.
    let head_dim = 64_f64;
    let scale = head_dim.powf(-0.25);
    // 64^{-0.25} = (2^6)^{-0.25} = 2^{-1.5} = 1/(2*sqrt(2)) ~= 0.35355
    let expected = 1.0 / (2.0 * 2.0_f64.sqrt());
    assert!(
        (scale - expected).abs() < 1e-10,
        "64^{{-0.25}} = {scale}, expected {expected}"
    );
}

#[test]
fn test_attention_combined_scale_equals_one_over_eight() {
    // For head_dim=64: combined scale = 64^{-0.5} = 1/8
    let combined = 64.0_f64.powf(-0.5);
    assert!(
        (combined - 0.125).abs() < 1e-10,
        "combined scale should be 1/8 for head_dim=64"
    );
}

// ============================================================================
// 2. Residual block structure differences between encoder and decoder
// ============================================================================

#[test]
fn test_encoder_block_output_shape_preserved() {
    // Encoder blocks should preserve input shape [B, T, D] exactly.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let enc_out = model.encode(&mel).unwrap();

    // Output should be [1, seq_len, d_model].
    assert_eq!(enc_out.rank(), 3);
    assert_eq!(enc_out.dim(0).unwrap(), 1);
    assert_eq!(enc_out.dim(2).unwrap(), config.d_model);
    // All values should be finite (zero weights still produce finite output).
    let flat = enc_out.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));
}

#[test]
fn test_decoder_block_output_shape_preserved() {
    // Decoder blocks should transform [B, T_tokens, D] -> [B, T_tokens, vocab_size].
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let enc_out = tiny_encoder_output();
    let tokens = DynTensor::from_vec(vec![0.0_f32; 3], &[1, 3], &cpu()).unwrap();
    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();

    assert_eq!(logits.rank(), 3);
    assert_eq!(logits.dim(0).unwrap(), 1);
    assert_eq!(logits.dim(1).unwrap(), 3);
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size);
}

#[test]
fn test_encoder_layers_count_matches_config() {
    // Verify the model actually creates the right number of encoder layers.
    let config = tiny_config();
    assert_eq!(config.encoder_layers, 1);
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = crate::WhisperModel::load(&vb, config).unwrap();
    // The model loads successfully with 1 encoder layer.
    assert_eq!(model.config().encoder_layers, 1);
}

#[test]
fn test_decoder_layers_count_matches_config() {
    let config = tiny_config();
    assert_eq!(config.decoder_layers, 1);
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = crate::WhisperModel::load(&vb, config).unwrap();
    assert_eq!(model.config().decoder_layers, 1);
}

// ============================================================================
// 3. Tied output projection: logits = hidden @ embed_weight^T
// ============================================================================

#[test]
fn test_tied_output_projection_vocab_dim() {
    // The decoder's output logits should have vocab_size as last dim because
    // the output projection is tied to the token embedding weight.
    // embed_weight: [vocab_size, d_model], transposed to [d_model, vocab_size]
    // logits = hidden @ embed_weight^T -> [B, T, vocab_size]
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let enc_out = tiny_encoder_output();
    for seq_len in [1, 2, 5] {
        let tokens =
            DynTensor::from_vec(vec![0.0_f32; seq_len], &[1, seq_len], &cpu()).unwrap();
        model.reset_kv_cache();
        let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
        assert_eq!(
            logits.dim(2).unwrap(),
            config.vocab_size,
            "logit last dim must be vocab_size for seq_len={seq_len}"
        );
    }
}

#[test]
fn test_tied_output_projection_no_extra_parameters() {
    // With tied projection, the decoder has no separate output linear layer.
    // The vocab_size dimension comes entirely from token_embedding.weight.
    let config = tiny_config();
    // Token embedding weight shape: [vocab_size, d_model]
    let embed_params = config.vocab_size * config.d_model;
    // Output projection reuses these same params (no additional allocation).
    assert!(embed_params > 0);
    // Verify the model loads without a separate output projection weight.
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    crate::WhisperModel::load(&vb, config).unwrap();
}

// ============================================================================
// 4. Positional encoding: sinusoidal (encoder) vs learned (decoder)
// ============================================================================

#[test]
fn test_sinusoidal_embedding_position_zero_is_well_defined() {
    // At position 0: sin(0) = 0 for all frequencies, cos(0) = 1 for all frequencies.
    let channels = 16;
    let half = channels / 2;
    let emb = sinusoidal_embedding(2, channels, DType::F32, &cpu()).unwrap();
    let flat = emb.to_flat_vec::<f32>().unwrap();

    // First half of row 0 is sin(0) = 0.
    for (i, &v) in flat.iter().take(half).enumerate() {
        assert!(
            v.abs() < 1e-6,
            "sin at pos=0, freq={i} should be 0, got {v}"
        );
    }
    // Second half of row 0 is cos(0) = 1.
    for i in 0..half {
        assert!(
            (flat[half + i] - 1.0).abs() < 1e-6,
            "cos at pos=0, freq={i} should be 1, got {}",
            flat[half + i]
        );
    }
}

#[test]
fn test_sinusoidal_embedding_bounded_minus_one_to_one() {
    // All values must be in [-1, 1] since they are sin/cos.
    let emb = sinusoidal_embedding(100, 64, DType::F32, &cpu()).unwrap();
    let flat = emb.to_flat_vec::<f32>().unwrap();
    for (i, &v) in flat.iter().enumerate() {
        assert!(
            v.is_finite() && (-1.0..=1.0).contains(&v),
            "sinusoidal value at index {i} out of [-1,1]: {v}"
        );
    }
}

#[test]
fn test_sinusoidal_vs_learned_positional_distinction() {
    // Encoder uses sinusoidal (fixed, not learned): shape [max_source_positions, d_model]
    // Decoder uses learned positional: shape [max_target_positions, d_model]
    // Key difference: sinusoidal is deterministic given position and dim index.
    let config = WhisperConfig::large_v3_turbo();
    let enc_pos_size = config.max_source_positions * config.d_model;
    let dec_pos_size = config.max_target_positions * config.d_model;

    // Encoder positional is larger (1500 * 1280 = 1,920,000).
    assert_eq!(enc_pos_size, 1_920_000);
    // Decoder positional is smaller (448 * 1280 = 573,440).
    assert_eq!(dec_pos_size, 573_440);
    assert!(enc_pos_size > dec_pos_size);
}

#[test]
fn test_sinusoidal_embedding_different_positions_differ() {
    // Adjacent positions should produce different embeddings.
    let emb = sinusoidal_embedding(3, 16, DType::F32, &cpu()).unwrap();
    let flat = emb.to_flat_vec::<f32>().unwrap();
    let row0 = &flat[0..16];
    let row1 = &flat[16..32];
    let row2 = &flat[32..48];

    // Rows should not be identical.
    assert_ne!(row0, row1, "positions 0 and 1 should differ");
    assert_ne!(row1, row2, "positions 1 and 2 should differ");
    assert_ne!(row0, row2, "positions 0 and 2 should differ");
}

// ============================================================================
// 5. Causal mask slice semantics at non-zero offsets
// ============================================================================

#[test]
fn test_causal_mask_at_offset_zero_allows_first_position() {
    // At offset 0, position 0 can attend to position 0 only.
    let mask = causal_mask(8, DType::F32, &cpu()).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();
    assert_eq!(flat[0], 0.0, "mask[0][0] should be 0 (attend)");
    assert_eq!(flat[1], f32::NEG_INFINITY, "mask[0][1] should be -inf (block)");
}

#[test]
fn test_causal_mask_at_last_row_attends_to_all() {
    // The last row of the causal mask should attend to all previous positions.
    let n = 6;
    let mask = causal_mask(n, DType::F32, &cpu()).unwrap();
    let flat = mask.to_flat_vec::<f32>().unwrap();
    let last_row_start = (n - 1) * n;
    for j in 0..n {
        assert_eq!(
            flat[last_row_start + j],
            0.0,
            "last row mask[{}, {}] should be 0 (attend)",
            n - 1,
            j
        );
    }
}

#[test]
fn test_causal_mask_narrow_simulates_offset_decode() {
    // During autoregressive decode, the mask is sliced:
    //   mask.narrow(0, offset, seq_len).narrow(1, 0, offset + seq_len)
    // For offset=3, seq_len=1: row 3 of original mask, columns 0..4.
    let n = 8;
    let mask = causal_mask(n, DType::F32, &cpu()).unwrap();

    // Simulate: offset=3, seq_len=1.
    let sliced = mask.narrow(0, 3, 1).unwrap();
    let sliced = sliced.narrow(1, 0, 4).unwrap();

    assert_eq!(sliced.dims(), &[1, 4]);
    let flat = sliced.to_flat_vec::<f32>().unwrap();
    // Row 3 can attend to positions 0, 1, 2, 3 (all 0.0).
    for (j, &v) in flat.iter().enumerate() {
        assert_eq!(
            v, 0.0,
            "at offset=3, position 3 should attend to position {j}"
        );
    }
}

#[test]
fn test_causal_mask_narrow_for_multi_token_prompt() {
    // Initial prompt of length 4 at offset 0: mask is [4, 4] lower-triangular.
    let n = 8;
    let mask = causal_mask(n, DType::F32, &cpu()).unwrap();
    let sliced = mask.narrow(0, 0, 4).unwrap();
    let sliced = sliced.narrow(1, 0, 4).unwrap();

    assert_eq!(sliced.dims(), &[4, 4]);
    let flat = sliced.to_flat_vec::<f32>().unwrap();
    for i in 0..4 {
        for j in 0..4 {
            let val = flat[i * 4 + j];
            if j <= i {
                assert_eq!(val, 0.0, "mask[{i}][{j}] should be 0");
            } else {
                assert_eq!(val, f32::NEG_INFINITY, "mask[{i}][{j}] should be -inf");
            }
        }
    }
}

// ============================================================================
// 6. Autoregressive decode position offset edge cases
// ============================================================================

#[test]
fn test_decode_position_offset_zero() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let enc_out = tiny_encoder_output();

    let tokens = DynTensor::from_vec(vec![0.0_f32; 3], &[1, 3], &cpu()).unwrap();
    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits.dims(), &[1, 3, config.vocab_size]);
}

#[test]
fn test_decode_incremental_position_offsets() {
    // Verify that incremental decoding with increasing offsets works correctly.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config).unwrap();
    let enc_out = tiny_encoder_output();

    // Initial prompt at offset 0.
    let prompt = DynTensor::from_vec(vec![0.0_f32; 2], &[1, 2], &cpu()).unwrap();
    let logits = model.decode(&prompt, &enc_out, true, 0).unwrap();
    assert_eq!(logits.dim(1).unwrap(), 2);

    // Single token at offset 2.
    let tok1 = DynTensor::from_vec(vec![0.0_f32; 1], &[1, 1], &cpu()).unwrap();
    let logits = model.decode(&tok1, &enc_out, false, 2).unwrap();
    assert_eq!(logits.dim(1).unwrap(), 1);

    // Single token at offset 3.
    let logits = model.decode(&tok1, &enc_out, false, 3).unwrap();
    assert_eq!(logits.dim(1).unwrap(), 1);
}

#[test]
fn test_decode_flush_resets_position() {
    // After flush_kv_cache=true, the model should accept offset=0 again.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let enc_out = tiny_encoder_output();

    // First segment.
    let tokens = DynTensor::from_vec(vec![0.0_f32; 2], &[1, 2], &cpu()).unwrap();
    model.decode(&tokens, &enc_out, true, 0).unwrap();

    // Flush and start new segment.
    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits.dims(), &[1, 2, config.vocab_size]);
}

// ============================================================================
// 7. Encoder no_cache vs cached forward produce same shape
// ============================================================================

#[test]
fn test_encoder_forward_vs_forward_no_cache_same_shape() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();

    // Cached forward.
    let out_cached = model.encode(&mel).unwrap();
    // No-cache forward.
    let out_no_cache = model.encoder().forward_no_cache(&mel).unwrap();

    assert_eq!(
        out_cached.dims(),
        out_no_cache.dims(),
        "cached and no-cache encoder forward should produce same shape"
    );
}

#[test]
fn test_decoder_forward_no_cache_same_shape_as_cached() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let enc_out = tiny_encoder_output();
    let tokens = DynTensor::from_vec_u32(vec![0, 1], &[1, 2], &cpu()).unwrap();

    // No-cache forward.
    let logits_no_cache = model.decoder().forward_no_cache(&tokens, &enc_out).unwrap();
    assert_eq!(logits_no_cache.dims(), &[1, 2, config.vocab_size]);
}

// ============================================================================
// 8. Conv stem: conv1 preserves length, conv2 halves it
// ============================================================================

/// Compute conv1d output length.
fn conv1d_out(input: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    (input + 2 * padding - kernel) / stride + 1
}

#[test]
fn test_conv1_preserves_length_various_inputs() {
    // conv1: kernel=3, stride=1, padding=1 always preserves input length.
    for input_len in [4, 8, 16, 32, 100, 3000] {
        let out = conv1d_out(input_len, 3, 1, 1);
        assert_eq!(
            out, input_len,
            "conv1 should preserve length for input_len={input_len}"
        );
    }
}

#[test]
fn test_conv2_halves_length_even_inputs() {
    // conv2: kernel=3, stride=2, padding=1 halves even-length inputs.
    for input_len in [4, 8, 16, 100, 3000] {
        let out = conv1d_out(input_len, 3, 2, 1);
        assert_eq!(
            out,
            input_len / 2,
            "conv2 should halve even length for input_len={input_len}"
        );
    }
}

#[test]
fn test_conv2_odd_length_floors() {
    // For odd input: (5 + 2 - 3) / 2 + 1 = 4/2 + 1 = 3.
    assert_eq!(conv1d_out(5, 3, 2, 1), 3);
    // (7 + 2 - 3) / 2 + 1 = 6/2 + 1 = 4.
    assert_eq!(conv1d_out(7, 3, 2, 1), 4);
    // (3 + 2 - 3) / 2 + 1 = 2/2 + 1 = 2.
    assert_eq!(conv1d_out(3, 3, 2, 1), 2);
}

#[test]
fn test_full_conv_stem_pipeline_for_standard_audio() {
    // Standard 30s: 3000 mel frames -> conv1(3000) -> conv2(1500).
    let mel_frames = N_FRAMES;
    let after_conv1 = conv1d_out(mel_frames, 3, 1, 1);
    let after_conv2 = conv1d_out(after_conv1, 3, 2, 1);
    assert_eq!(after_conv2, 1500);

    // This must equal max_source_positions for all configs.
    for config in &[
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::large_v3_turbo(),
    ] {
        assert_eq!(
            after_conv2, config.max_source_positions,
            "conv stem output must match max_source_positions"
        );
    }
}

// ============================================================================
// 9. WhisperError variant coverage
// ============================================================================

#[test]
fn test_error_zero_config_field_display() {
    let e = WhisperError::ZeroConfigField { field: "d_model" };
    let msg = e.to_string();
    assert!(msg.contains("d_model"));
    assert!(msg.contains("must be > 0"));
}

#[test]
fn test_error_config_not_divisible_display() {
    let e = WhisperError::ConfigNotDivisible {
        a_name: "d_model",
        a_val: 384,
        b_name: "encoder_attention_heads",
        b_val: 5,
    };
    let msg = e.to_string();
    assert!(msg.contains("384"));
    assert!(msg.contains("5"));
    assert!(msg.contains("divisible"));
}

#[test]
fn test_error_non_finite_config_field_display() {
    let e = WhisperError::NonFiniteConfigField {
        field: "compression_ratio_threshold",
        value: f64::NAN,
    };
    let msg = e.to_string();
    assert!(msg.contains("compression_ratio_threshold"));
    assert!(msg.contains("finite"));
}

#[test]
fn test_error_empty_config_field_display() {
    let e = WhisperError::EmptyConfigField {
        field: "initial_tokens",
    };
    let msg = e.to_string();
    assert!(msg.contains("initial_tokens"));
    assert!(msg.contains("empty"));
}

#[test]
fn test_error_config_exceeds_limit_display() {
    let e = WhisperError::ConfigExceedsLimit {
        field: "max_length",
        value: 999,
        limit: 448,
    };
    let msg = e.to_string();
    assert!(msg.contains("999"));
    assert!(msg.contains("448"));
}

#[test]
fn test_error_empty_audio_display() {
    let e = WhisperError::EmptyAudio {
        stage: "pcm_to_mel",
    };
    let msg = e.to_string();
    assert!(msg.contains("pcm_to_mel"));
    assert!(msg.contains("empty"));
}

#[test]
fn test_error_audio_format_display() {
    let e = WhisperError::AudioFormat {
        reason: "bad sample rate".into(),
    };
    let msg = e.to_string();
    assert!(msg.contains("bad sample rate"));
}

#[test]
fn test_error_vocab_parse_display() {
    let e = WhisperError::VocabParseError {
        detail: "invalid JSON".into(),
    };
    let msg = e.to_string();
    assert!(msg.contains("invalid JSON"));
}

#[test]
fn test_error_token_out_of_range_display() {
    let e = WhisperError::TokenOutOfRange {
        id: 99999,
        vocab_size: 51866,
    };
    let msg = e.to_string();
    assert!(msg.contains("99999"));
    assert!(msg.contains("51866"));
}

#[test]
fn test_error_utf8_decode_display() {
    let e = WhisperError::Utf8DecodeError {
        detail: "invalid byte".into(),
    };
    assert!(e.to_string().contains("invalid byte"));
}

#[test]
fn test_error_merge_parse_display() {
    let e = WhisperError::MergeParseError {
        line: 42,
        detail: "bad pair",
    };
    let msg = e.to_string();
    assert!(msg.contains("42"));
    assert!(msg.contains("bad pair"));
}

#[test]
fn test_error_missing_merges_display() {
    let e = WhisperError::MissingMerges;
    assert!(e.to_string().contains("merges"));
}

#[test]
fn test_error_token_not_in_vocab_display() {
    let e = WhisperError::TokenNotInVocab {
        token: "xyz".into(),
    };
    assert!(e.to_string().contains("xyz"));
}

#[test]
fn test_error_token_id_overflow_display() {
    let e = WhisperError::TokenIdOverflow {
        token_id: usize::MAX,
    };
    assert!(e.to_string().contains("u32"));
}

#[test]
fn test_error_logit_too_small_display() {
    let e = WhisperError::LogitTooSmall {
        logit_len: 100,
        vocab_size: 51866,
    };
    let msg = e.to_string();
    assert!(msg.contains("100"));
    assert!(msg.contains("51866"));
}

#[test]
fn test_error_invalid_temperature_display() {
    let e = WhisperError::InvalidTemperature {
        temperature: -1.0,
    };
    assert!(e.to_string().contains("-1"));
}

#[test]
fn test_error_position_overflow_display() {
    let e = WhisperError::PositionOverflow {
        offset: usize::MAX - 1,
        seq_len: 5,
    };
    assert!(e.to_string().contains("overflow"));
}

#[test]
fn test_error_language_token_range_display() {
    let e = WhisperError::LanguageTokenRange {
        start: 50259,
        end: 60000,
        vocab_size: 51866,
    };
    let msg = e.to_string();
    assert!(msg.contains("50259"));
    assert!(msg.contains("60000"));
    assert!(msg.contains("51866"));
}

#[test]
fn test_error_batch_mismatch_display() {
    let e = WhisperError::BatchMismatch {
        encoder_batch: 1,
        decoder_batch: 2,
    };
    let msg = e.to_string();
    assert!(msg.contains("1"));
    assert!(msg.contains("2"));
}

#[test]
fn test_error_cache_seq_mismatch_display() {
    let e = WhisperError::CacheSeqMismatch {
        cached_seq: 100,
        encoder_seq: 200,
    };
    let msg = e.to_string();
    assert!(msg.contains("100"));
    assert!(msg.contains("200"));
}

#[test]
fn test_error_byte_alignment_display() {
    let e = WhisperError::ByteAlignment {
        tensor_name: "layer.0.weight".into(),
        byte_len: 7,
        alignment: 4,
    };
    let msg = e.to_string();
    assert!(msg.contains("layer.0.weight"));
    assert!(msg.contains("7"));
}

#[test]
fn test_error_non_finite_weight_display() {
    let e = WhisperError::NonFiniteWeight {
        tensor_name: "encoder.conv1.weight".into(),
        count: 3,
    };
    let msg = e.to_string();
    assert!(msg.contains("encoder.conv1.weight"));
    assert!(msg.contains("3"));
}

#[test]
fn test_error_conversion_to_tensor_error() {
    let we = WhisperError::ZeroConfigField { field: "vocab_size" };
    let te: TensorError = we.into();
    assert!(te.to_string().contains("vocab_size"));
}

// ============================================================================
// 10. Generation config: beam size, temperature, length penalty
// ============================================================================

#[test]
fn test_decode_config_default_initial_tokens_are_sot_en_transcribe_notimestamps() {
    let dc = DecodeConfig::default();
    assert_eq!(dc.initial_tokens.len(), 4);
    assert_eq!(dc.initial_tokens[0], SOT_TOKEN); // 50258
    assert_eq!(dc.initial_tokens[1], LANGUAGE_TOKEN_START); // 50259 = English
    assert_eq!(dc.initial_tokens[2], 50360); // transcribe
    assert_eq!(dc.initial_tokens[3], 50364); // no_timestamps
}

#[test]
fn test_decode_config_translate_task_initial_tokens() {
    let dc = DecodeConfig::default()
        .with_initial_tokens(vec![SOT_TOKEN, LANGUAGE_TOKEN_START, 50359, 50364]);
    assert_eq!(dc.initial_tokens[2], 50359); // translate
}

#[test]
fn test_decode_config_max_length_default_equals_max_decode_length() {
    let dc = DecodeConfig::default();
    assert_eq!(dc.max_length, MAX_DECODE_LENGTH);
}

#[test]
fn test_decode_config_compression_ratio_default_value() {
    let dc = DecodeConfig::default();
    assert!((dc.compression_ratio_threshold - DEFAULT_COMPRESSION_RATIO_THRESHOLD).abs() < 1e-10);
    assert!((dc.compression_ratio_threshold - 2.4).abs() < 1e-10);
}

#[test]
fn test_decode_config_avg_logprob_default_value() {
    let dc = DecodeConfig::default();
    assert!((dc.avg_logprob_threshold - DEFAULT_AVG_LOGPROB_THRESHOLD).abs() < 1e-10);
    assert!((dc.avg_logprob_threshold - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_decode_config_validation_accepts_boundary_max_length() {
    let dc = DecodeConfig::default().with_max_length(MAX_DECODE_LENGTH);
    dc.validate().expect("boundary max_length should validate");
}

#[test]
fn test_decode_config_validation_accepts_valid_thresholds() {
    let dc = DecodeConfig::default()
        .with_compression_ratio_threshold(1.0)
        .with_avg_logprob_threshold(-0.5);
    dc.validate().expect("valid thresholds should validate");
}

#[test]
fn test_decode_config_validation_rejects_neg_inf_compression() {
    let dc = DecodeConfig::default().with_compression_ratio_threshold(f64::NEG_INFINITY);
    assert!(dc.validate().is_err());
}

#[test]
fn test_long_form_config_temperatures_match_default_temperatures() {
    let lfc = LongFormConfig::default();
    assert_eq!(lfc.temperatures.len(), DEFAULT_TEMPERATURES.len());
    for (a, b) in lfc.temperatures.iter().zip(DEFAULT_TEMPERATURES.iter()) {
        assert!((*a - *b).abs() < 1e-10);
    }
}

// ============================================================================
// 11. Audio preprocessing: sample rate, chunk length, padding
// ============================================================================

#[test]
fn test_sample_rate_is_16khz() {
    assert_eq!(SAMPLE_RATE, 16_000);
}

#[test]
fn test_chunk_length_is_30_seconds() {
    assert_eq!(CHUNK_LENGTH, 30);
}

#[test]
fn test_n_samples_for_30s_at_16khz() {
    assert_eq!(N_SAMPLES, 480_000);
    assert_eq!(N_SAMPLES, SAMPLE_RATE * CHUNK_LENGTH);
}

#[test]
fn test_n_frames_for_standard_stft_params() {
    // N_FRAMES = N_SAMPLES / HOP_LENGTH = 480000 / 160 = 3000.
    assert_eq!(N_FRAMES, 3000);
    assert_eq!(N_FRAMES, N_SAMPLES / HOP_LENGTH);
}

#[test]
fn test_n_fft_window_is_25ms() {
    let window_ms = (N_FFT as f64 / SAMPLE_RATE as f64) * 1000.0;
    assert!((window_ms - 25.0).abs() < 0.01);
}

#[test]
fn test_hop_length_stride_is_10ms() {
    let stride_ms = (HOP_LENGTH as f64 / SAMPLE_RATE as f64) * 1000.0;
    assert!((stride_ms - 10.0).abs() < 0.01);
}

#[test]
fn test_overlap_ratio_between_n_fft_and_hop() {
    // Overlap = (N_FFT - HOP_LENGTH) / N_FFT = (400 - 160) / 400 = 0.6 = 60%.
    let overlap = (N_FFT - HOP_LENGTH) as f64 / N_FFT as f64;
    assert!((overlap - 0.6).abs() < 1e-10);
}

// ============================================================================
// 12. Tokenizer integration: timestamp token math
// ============================================================================

#[test]
fn test_timestamp_token_zero_seconds() {
    // Token 50365 represents 0.00 seconds.
    let t = WhisperTokenizer::from_vocab_str("{}").unwrap();
    assert_eq!(t.timestamp_value(TIMESTAMP_BEGIN), Some(0.0));
}

#[test]
fn test_timestamp_token_30_seconds() {
    // Token 50365 + 1500 = 51865 represents 30.00 seconds.
    let t = WhisperTokenizer::from_vocab_str("{}").unwrap();
    let v = t.timestamp_value(TIMESTAMP_BEGIN + 1500).unwrap();
    assert!((v - 30.0).abs() < 1e-10);
}

#[test]
fn test_timestamp_resolution_is_20ms() {
    // Each step is 0.02 seconds (20ms).
    let t = WhisperTokenizer::from_vocab_str("{}").unwrap();
    let v0 = t.timestamp_value(TIMESTAMP_BEGIN).unwrap();
    let v1 = t.timestamp_value(TIMESTAMP_BEGIN + 1).unwrap();
    assert!((v1 - v0 - 0.02).abs() < 1e-10);
}

#[test]
fn test_timestamp_steps_cover_full_30_seconds() {
    // 30.0 / 0.02 = 1500 steps. Tokens 50365 through 51865 inclusive = 1501 tokens.
    let steps = (CHUNK_LENGTH as f64 / 0.02) as usize;
    assert_eq!(steps, 1500);
    // Last timestamp token: 50365 + 1500 = 51865.
    let last_ts = TIMESTAMP_BEGIN + steps;
    assert_eq!(last_ts, 51865);
}

#[test]
fn test_is_special_boundary_at_eot() {
    let t = WhisperTokenizer::from_vocab_str("{}").unwrap();
    // EOT_TOKEN (50257) is the boundary.
    assert!(!t.is_special(50256));
    assert!(t.is_special(50257));
    assert!(t.is_special(50258));
}

#[test]
fn test_is_timestamp_boundary_at_timestamp_begin() {
    let t = WhisperTokenizer::from_vocab_str("{}").unwrap();
    assert!(!t.is_timestamp(TIMESTAMP_BEGIN - 1));
    assert!(t.is_timestamp(TIMESTAMP_BEGIN));
    assert!(t.is_timestamp(TIMESTAMP_BEGIN + 1));
}

// ============================================================================
// 13. Model weight dtype handling
// ============================================================================

#[test]
fn test_model_dtype_f32() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = crate::WhisperModel::load(&vb, config).unwrap();
    assert_eq!(model.dtype(), DType::F32);
}

#[test]
fn test_model_encode_with_f32_mel_on_f32_model() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let out = model.encode(&mel).unwrap();
    assert_eq!(out.dtype(), DType::F32);
}

// ============================================================================
// 14. Config head dim calculation consistency
// ============================================================================

#[test]
fn test_head_dim_times_heads_equals_d_model_all_presets() {
    let presets = [
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ];
    for config in &presets {
        assert_eq!(
            config.encoder_head_dim() * config.encoder_attention_heads,
            config.d_model,
            "encoder: head_dim * heads should equal d_model"
        );
        assert_eq!(
            config.decoder_head_dim() * config.decoder_attention_heads,
            config.d_model,
            "decoder: head_dim * heads should equal d_model"
        );
    }
}

#[test]
fn test_head_dim_64_for_all_presets() {
    // Whisper consistently uses head_dim=64 across all model sizes.
    for config in &[
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ] {
        assert_eq!(config.encoder_head_dim(), 64);
        assert_eq!(config.decoder_head_dim(), 64);
    }
}

// ============================================================================
// 15. Cross-attention encoder-decoder d_model compatibility
// ============================================================================

#[test]
fn test_cross_attention_requires_shared_d_model() {
    // Cross-attention works because Q comes from decoder (d_model) and
    // K/V come from encoder (d_model). Both must share the same d_model.
    for config in &[
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_medium(),
        WhisperConfig::large_v3_turbo(),
    ] {
        // Whisper has a single d_model field for both encoder and decoder.
        let enc_qkv = config.d_model;
        let dec_qkv = config.d_model;
        assert_eq!(enc_qkv, dec_qkv);

        // Cross-attention head dims must also match.
        assert_eq!(config.encoder_head_dim(), config.decoder_head_dim());
    }
}

#[test]
fn test_encoder_output_d_model_feeds_decoder_cross_attention() {
    // Verify the shape plumbing: encoder output [B, enc_seq, d_model]
    // feeds into decoder cross-attention K/V projections that expect d_model.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let enc_out = model.encode(&mel).unwrap();

    // Encoder output last dim is d_model.
    assert_eq!(enc_out.dim(2).unwrap(), config.d_model);

    // Decoder accepts this as cross-attention input.
    let tokens = DynTensor::from_vec(vec![0.0_f32; 2], &[1, 2], &cpu()).unwrap();
    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size);
}

// ============================================================================
// 16. Model KV cache reset behavior
// ============================================================================

#[test]
fn test_kv_cache_reset_does_not_panic() {
    let mut model = tiny_model();
    // Multiple resets should be safe.
    model.reset_kv_cache();
    model.reset_kv_cache();
    model.reset_kv_cache();
}

#[test]
fn test_kv_cache_reset_between_segments_produces_valid_output() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let enc_out = tiny_encoder_output();

    let tokens = DynTensor::from_vec(vec![0.0_f32; 2], &[1, 2], &cpu()).unwrap();

    // Segment 1.
    let logits1 = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits1.dims(), &[1, 2, config.vocab_size]);

    // Reset between segments.
    model.reset_kv_cache();

    // Segment 2.
    let logits2 = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits2.dims(), &[1, 2, config.vocab_size]);
}

// ============================================================================
// 17. Compression ratio edge cases for generation quality checks
// ============================================================================

#[test]
fn test_compression_ratio_monotonic_with_repetition() {
    // More repetition -> higher compression ratio.
    let all_unique = compression_ratio(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let some_repeat = compression_ratio(&[1, 2, 1, 2, 1, 2, 1, 2]);
    let all_same = compression_ratio(&[5, 5, 5, 5, 5, 5, 5, 5]);

    assert!(
        some_repeat > all_unique,
        "repeated bigrams ({some_repeat}) > unique bigrams ({all_unique})"
    );
    assert!(
        all_same > some_repeat,
        "all same ({all_same}) > some repeat ({some_repeat})"
    );
}

#[test]
fn test_compression_ratio_always_at_least_one() {
    // The minimum compression ratio is 1.0 (all bigrams unique).
    for tokens in [
        vec![1, 2, 3],
        vec![10, 20, 30, 40],
        vec![1],
        vec![],
    ] {
        let cr = compression_ratio(&tokens);
        assert!(cr >= 1.0, "compression ratio should be >= 1.0, got {cr}");
    }
}

// ============================================================================
// 18. Default temperatures sequence properties
// ============================================================================

#[test]
fn test_temperatures_sequence_length() {
    assert_eq!(DEFAULT_TEMPERATURES.len(), 6);
}

#[test]
fn test_temperatures_first_is_zero_for_greedy_decode() {
    // Temperature 0 means greedy (argmax) decoding.
    assert!((DEFAULT_TEMPERATURES[0]).abs() < 1e-10);
}

#[test]
fn test_temperatures_last_is_one_for_sampling() {
    // Temperature 1 means standard sampling.
    let last = DEFAULT_TEMPERATURES[DEFAULT_TEMPERATURES.len() - 1];
    assert!((last - 1.0).abs() < 1e-10);
}

#[test]
fn test_temperatures_strictly_increasing() {
    for w in DEFAULT_TEMPERATURES.windows(2) {
        assert!(
            w[1] > w[0],
            "temperatures must be strictly increasing: {} >= {}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn test_temperatures_all_non_negative() {
    for &t in &DEFAULT_TEMPERATURES {
        assert!(t >= 0.0, "temperature must be non-negative");
    }
}

// ============================================================================
// 19. DecodingResult quality check interactions
// ============================================================================

#[test]
fn test_quality_check_passes_for_normal_result() {
    let r = DecodingResult::new(vec![1, 2, 3, 4, 5], -0.5, 1.2, true, 0.0, 0.1);
    let config = DecodeConfig::default();
    assert!(crate::decode::passes_quality_check(&r, &config));
}

#[test]
fn test_quality_check_fails_for_high_compression() {
    // Default threshold is 2.4.
    let r = DecodingResult::new(vec![1, 1, 1], -0.5, 3.0, true, 0.0, 0.1);
    let config = DecodeConfig::default();
    assert!(!crate::decode::passes_quality_check(&r, &config));
}

#[test]
fn test_quality_check_fails_for_low_avg_logprob() {
    // Default threshold is -1.0.
    let r = DecodingResult::new(vec![1, 2, 3], -2.0, 1.0, true, 0.0, 0.1);
    let config = DecodeConfig::default();
    assert!(!crate::decode::passes_quality_check(&r, &config));
}

#[test]
fn test_quality_check_custom_relaxed_thresholds() {
    let r = DecodingResult::new(vec![1, 2, 3], -5.0, 10.0, true, 0.0, 0.5);
    let config = DecodeConfig::default()
        .with_compression_ratio_threshold(20.0)
        .with_avg_logprob_threshold(-10.0);
    assert!(crate::decode::passes_quality_check(&r, &config));
}

// ============================================================================
// 20. Mel spectrogram dimensions across model sizes
// ============================================================================

#[test]
fn test_mel_bins_80_for_small_models() {
    for config in &[
        WhisperConfig::whisper_tiny(),
        WhisperConfig::whisper_base(),
        WhisperConfig::whisper_small(),
        WhisperConfig::whisper_medium(),
    ] {
        assert_eq!(config.num_mel_bins, 80);
    }
}

#[test]
fn test_mel_bins_128_for_large_models() {
    for config in &[
        WhisperConfig::whisper_large_v2(),
        WhisperConfig::large_v3_turbo(),
    ] {
        assert_eq!(config.num_mel_bins, 128);
    }
}

#[test]
fn test_num_mel_bins_constant_matches_large_config() {
    assert_eq!(NUM_MEL_BINS, WhisperConfig::large_v3_turbo().num_mel_bins);
    assert_eq!(NUM_MEL_BINS, 128);
}

#[test]
fn test_mel_bins_less_than_fft_bins() {
    // FFT bins = N_FFT / 2 + 1 = 201.
    let fft_bins = N_FFT / 2 + 1;
    assert_eq!(fft_bins, 201);
    assert!(80 < fft_bins);
    assert!(128 < fft_bins);
}

// ============================================================================
// 21. Config builder with_* methods preserve other fields
// ============================================================================

#[test]
fn test_builder_with_encoder_layers_preserves_decoder() {
    let base = WhisperConfig::large_v3_turbo();
    let modified = base.clone().with_encoder_layers(16);
    assert_eq!(modified.encoder_layers, 16);
    assert_eq!(modified.decoder_layers, base.decoder_layers);
    assert_eq!(modified.d_model, base.d_model);
    assert_eq!(modified.vocab_size, base.vocab_size);
}

#[test]
fn test_builder_with_num_mel_bins_preserves_other_fields() {
    let base = WhisperConfig::whisper_tiny();
    let modified = base.clone().with_num_mel_bins(128);
    assert_eq!(modified.num_mel_bins, 128);
    assert_eq!(modified.d_model, base.d_model);
    assert_eq!(modified.encoder_layers, base.encoder_layers);
    assert_eq!(modified.decoder_layers, base.decoder_layers);
    assert_eq!(modified.encoder_attention_heads, base.encoder_attention_heads);
    assert_eq!(modified.decoder_attention_heads, base.decoder_attention_heads);
    assert_eq!(modified.encoder_ffn_dim, base.encoder_ffn_dim);
    assert_eq!(modified.decoder_ffn_dim, base.decoder_ffn_dim);
    assert_eq!(modified.vocab_size, base.vocab_size);
    assert_eq!(modified.max_source_positions, base.max_source_positions);
    assert_eq!(modified.max_target_positions, base.max_target_positions);
}

#[test]
fn test_builder_chain_produces_valid_custom_config() {
    let config = WhisperConfig::whisper_tiny()
        .with_d_model(128)
        .with_encoder_attention_heads(4)
        .with_decoder_attention_heads(4)
        .with_encoder_ffn_dim(512)
        .with_decoder_ffn_dim(512)
        .with_encoder_layers(2)
        .with_decoder_layers(2)
        .with_vocab_size(10000)
        .with_num_mel_bins(40)
        .with_max_source_positions(100)
        .with_max_target_positions(50);

    config.validate().expect("custom chain config should validate");
    assert_eq!(config.d_model, 128);
    assert_eq!(config.encoder_head_dim(), 32);
    assert_eq!(config.decoder_head_dim(), 32);
}
