// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit and integration tests for nn-models public APIs.
//!
//! Targets: config validation, STFT/iSTFT params, Silero VAD constants,
//! Kokoro config/error edge cases, streaming types, convert config,
//! error display/conversion, and signal processing constants.
//!
//! Part of #3809.

use nn_models::*;

// ============================================================================
// IstftParams validation
// ============================================================================

#[test]
fn test_istft_params_new_valid() {
    let p = IstftParams::new(8, 4, false, false).expect("valid params");
    assert_eq!(p.n_fft, 8);
    assert_eq!(p.hop_length, 4);
    assert!(!p.normalized);
    assert!(!p.center);
}

#[test]
fn test_istft_params_new_rejects_zero_nfft() {
    let err = IstftParams::new(0, 4, false, false).unwrap_err();
    assert!(
        matches!(err, IstftError::OddNfft { n_fft: 0 }),
        "zero n_fft should be rejected: {err:?}"
    );
}

#[test]
fn test_istft_params_new_rejects_odd_nfft() {
    let err = IstftParams::new(7, 3, false, false).unwrap_err();
    assert!(
        matches!(err, IstftError::OddNfft { n_fft: 7 }),
        "odd n_fft should be rejected: {err:?}"
    );
}

#[test]
fn test_istft_params_new_rejects_zero_hop() {
    let err = IstftParams::new(8, 0, false, false).unwrap_err();
    assert!(
        matches!(err, IstftError::ZeroHopLength),
        "zero hop should be rejected: {err:?}"
    );
}

#[test]
fn test_istft_params_default_htdemucs() {
    let p = IstftParams::default();
    assert_eq!(p.n_fft, 4096);
    assert_eq!(p.hop_length, 1024);
    assert!(p.normalized);
    assert!(p.center);
}

// ============================================================================
// IstftError display
// ============================================================================

#[test]
fn test_istft_error_odd_nfft_display() {
    let err = IstftError::OddNfft { n_fft: 5 };
    let msg = err.to_string();
    assert!(msg.contains("5"), "should contain n_fft value: {msg}");
    assert!(msg.contains("even"), "should mention even: {msg}");
}

#[test]
fn test_istft_error_zero_hop_display() {
    let err = IstftError::ZeroHopLength;
    let msg = err.to_string();
    assert!(
        msg.contains("hop_length"),
        "should mention hop_length: {msg}"
    );
}

#[test]
fn test_istft_error_converts_to_tensor_error() {
    let err = IstftError::OddNfft { n_fft: 3 };
    let te: nn_core::TensorError = err.into();
    // IstftError should convert to TensorError via From impl
    let msg = te.to_string();
    assert!(
        msg.contains("3") || msg.contains("odd") || msg.contains("even"),
        "tensor error should contain original message: {msg}"
    );
}

// ============================================================================
// StftParams construction and defaults
// ============================================================================

#[test]
fn test_stft_params_default_silero_vad() {
    let p = StftParams::default();
    assert_eq!(p.n_fft, 256);
    assert_eq!(p.hop_length, 128);
    assert_eq!(p.n_freqs, 129);
    assert_eq!(p.pad_right, 64);
}

#[test]
fn test_stft_params_n_freqs_formula() {
    // For any n_fft, n_freqs should be n_fft/2 + 1
    for n_fft in [4, 8, 16, 64, 256, 512, 1024, 4096] {
        let p = StftParams::new(n_fft, n_fft / 2);
        assert_eq!(
            p.n_freqs,
            n_fft / 2 + 1,
            "n_freqs formula for n_fft={n_fft}"
        );
    }
}

#[test]
fn test_stft_params_pad_right_formula() {
    let p = StftParams::new(512, 256);
    assert_eq!(p.pad_right, 128, "pad_right = n_fft / 4");
}

// ============================================================================
// StftError display
// ============================================================================

#[test]
fn test_stft_error_basis_size_mismatch_display() {
    let err = StftError::BasisSizeMismatch {
        expected: 66048,
        actual: 100,
    };
    let msg = err.to_string();
    assert!(msg.contains("66048"), "should show expected: {msg}");
    assert!(msg.contains("100"), "should show actual: {msg}");
}

#[test]
fn test_stft_error_freqs_mismatch_display() {
    let err = StftError::FreqsMismatch {
        expected: 129,
        actual: 200,
    };
    let msg = err.to_string();
    assert!(msg.contains("129"), "should show expected: {msg}");
    assert!(msg.contains("200"), "should show actual: {msg}");
}

#[test]
fn test_stft_error_converts_to_tensor_error() {
    let err = StftError::AudioTooShort {
        padded_len: 10,
        n_fft: 256,
    };
    let te: nn_core::TensorError = err.into();
    let msg = te.to_string();
    assert!(
        msg.contains("10") || msg.contains("256"),
        "should preserve error details: {msg}"
    );
}

// ============================================================================
// KokoroConfig
// ============================================================================

#[test]
fn test_kokoro_config_new_equals_default() {
    let new = KokoroConfig::new();
    let def = KokoroConfig::default();
    assert_eq!(new.d_en, def.d_en);
    assert_eq!(new.style_dim, def.style_dim);
    assert_eq!(new.n_fft, def.n_fft);
    assert_eq!(new.max_dur, def.max_dur);
    assert_eq!(new.gen_initial_channels, def.gen_initial_channels);
}

#[test]
fn test_kokoro_config_default_validates() {
    let cfg = KokoroConfig::default();
    cfg.validate().expect("default config should be valid");
}

#[test]
fn test_kokoro_config_upsample_product_is_hop_length() {
    let cfg = KokoroConfig::default();
    let product: usize = cfg.upsample_rates.iter().product();
    // [10, 6] -> 60 = the hop length for iSTFT
    assert_eq!(product, 60, "upsample rates product = hop length");
}

#[test]
fn test_kokoro_config_default_n_fft_is_20() {
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.n_fft, 20, "Kokoro default n_fft = 20");
    assert!(cfg.validate().is_ok(), "default config must be valid");
}

#[test]
fn test_kokoro_config_default_f0_bilstm_hidden() {
    let cfg = KokoroConfig::default();
    assert_eq!(cfg.f0_bilstm_hidden, 256);
}

// ============================================================================
// KokoroError
// ============================================================================

#[test]
fn test_kokoro_error_invalid_config_display() {
    let err = KokoroError::InvalidConfig {
        field: "test_field",
        reason: "test reason".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("test_field"), "should contain field: {msg}");
    assert!(msg.contains("test reason"), "should contain reason: {msg}");
}

#[test]
fn test_kokoro_error_invalid_input_display() {
    let err = KokoroError::InvalidInput("bad input data".into());
    let msg = err.to_string();
    assert!(
        msg.contains("bad input data"),
        "should contain message: {msg}"
    );
}

// ============================================================================
// KokoroStreamConfig
// ============================================================================

#[test]
fn test_stream_config_default_crossfade() {
    let cfg = KokoroStreamConfig::default();
    assert_eq!(
        cfg.crossfade_samples, 960,
        "default 40ms at 24kHz = 960 samples"
    );
    assert_eq!(cfg.crossfade_window, CrossfadeWindow::SqrtHann);
}

#[test]
fn test_stream_config_new_validates() {
    let cfg = KokoroStreamConfig::new(240).expect("valid config");
    assert_eq!(cfg.crossfade_samples, 240);
}

#[test]
fn test_stream_config_new_rejects_zero() {
    let err = KokoroStreamConfig::new(0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("crossfade_samples"),
        "should mention field: {msg}"
    );
}

#[test]
fn test_stream_config_with_window() {
    let cfg = KokoroStreamConfig::new(480)
        .unwrap()
        .with_window(CrossfadeWindow::Hann);
    assert_eq!(cfg.crossfade_window, CrossfadeWindow::Hann);
}

#[test]
fn test_stream_config_crossfade_duration_secs() {
    let cfg = KokoroStreamConfig::default();
    // 960 / 24000 = 0.04 seconds
    let dur = cfg.crossfade_duration_secs();
    assert!((dur - 0.04).abs() < 1e-9, "expected 0.04s, got {dur}");
}

// ============================================================================
// AudioChunk
// ============================================================================

#[test]
fn test_audio_chunk_new_and_accessors() {
    let pcm = vec![0.1, 0.2, 0.3, 0.4];
    let chunk = AudioChunk::new(pcm, 1, 0, 0, 3, false);
    assert_eq!(chunk.len(), 4);
    assert!(!chunk.is_empty());
    assert!(!chunk.is_final);
    assert_eq!(chunk.chunk_index, 0);
    assert_eq!(chunk.total_chunks, 3);
    assert_eq!(chunk.sample_offset, 0);
    assert_eq!(chunk.channels, 1);
}

#[test]
fn test_audio_chunk_duration_mono() {
    // 24000 samples at 24kHz = 1.0 second
    let pcm = vec![0.0; 24000];
    let chunk = AudioChunk::new(pcm, 1, 0, 0, 1, true);
    let dur = chunk.duration_secs();
    assert!(
        (dur - 1.0).abs() < 1e-9,
        "expected 1.0s for 24000 mono samples, got {dur}"
    );
}

#[test]
fn test_audio_chunk_duration_stereo() {
    // 48000 floats, 2 channels = 24000 frames = 1.0 second at 24kHz
    let pcm = vec![0.0; 48000];
    let chunk = AudioChunk::new(pcm, 2, 0, 0, 1, true);
    let dur = chunk.duration_secs();
    assert!(
        (dur - 1.0).abs() < 1e-9,
        "expected 1.0s for 48000 stereo samples, got {dur}"
    );
}

#[test]
fn test_audio_chunk_empty() {
    let chunk = AudioChunk::new(vec![], 1, 0, 0, 1, true);
    assert!(chunk.is_empty());
    assert_eq!(chunk.len(), 0);
    assert!((chunk.duration_secs() - 0.0).abs() < 1e-12);
}

#[test]
fn test_concatenate_chunks_basic() {
    let c1 = AudioChunk::new(vec![1.0, 2.0], 1, 0, 0, 2, false);
    let c2 = AudioChunk::new(vec![3.0, 4.0], 1, 2, 1, 2, true);
    let result = concatenate_chunks(&[c1, c2]);
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_concatenate_chunks_empty_input() {
    let result = concatenate_chunks(&[]);
    assert!(result.is_empty());
}

// ============================================================================
// CrossfadeWindow
// ============================================================================

#[test]
fn test_crossfade_window_default_is_sqrt_hann() {
    let w = CrossfadeWindow::default();
    assert_eq!(w, CrossfadeWindow::SqrtHann);
}

#[test]
fn test_crossfade_window_variants_not_equal() {
    assert_ne!(CrossfadeWindow::Linear, CrossfadeWindow::Hann);
    assert_ne!(CrossfadeWindow::Hann, CrossfadeWindow::SqrtHann);
    assert_ne!(CrossfadeWindow::Linear, CrossfadeWindow::SqrtHann);
}

// ============================================================================
// Silero VAD builder constants
// ============================================================================

#[test]
fn test_silero_encoder_blocks_channel_chain() {
    // Verify the encoder blocks form a valid channel chain:
    // each block's output channels == next block's input channels
    use nn_models::silero_vad_builders::ENCODER_BLOCKS;
    for i in 0..ENCODER_BLOCKS.len() - 1 {
        assert_eq!(
            ENCODER_BLOCKS[i].out_channels,
            ENCODER_BLOCKS[i + 1].in_channels,
            "block {i} out_channels should match block {} in_channels",
            i + 1
        );
    }
}

#[test]
fn test_silero_lstm_hidden_matches_last_encoder_output() {
    use nn_models::silero_vad_builders::{ENCODER_BLOCKS, LSTM_HIDDEN_SIZE};
    let last_block = &ENCODER_BLOCKS[ENCODER_BLOCKS.len() - 1];
    assert_eq!(
        last_block.out_channels, LSTM_HIDDEN_SIZE,
        "last encoder block output should match LSTM hidden size"
    );
}

#[test]
fn test_silero_encoder_blocks_all_kernel_size_3() {
    use nn_models::silero_vad_builders::ENCODER_BLOCKS;
    for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
        assert_eq!(block.kernel_size, 3, "block {i} should have kernel_size=3");
    }
}

// ============================================================================
// KokoroTokenizer
// ============================================================================

#[test]
fn test_tokenizer_max_tokens_constant() {
    assert_eq!(MAX_PHONEME_TOKENS, 510, "PlBert 512 - 2 padding = 510");
}

#[test]
fn test_tokenizer_pad_token_id() {
    assert_eq!(PAD_TOKEN_ID, 0);
}

#[test]
fn test_tokenizer_default_max_tokens() {
    let tok = KokoroTokenizer::kokoro_default();
    assert_eq!(tok.max_tokens(), MAX_PHONEME_TOKENS);
}

// ============================================================================
// ConvertConfig
// ============================================================================

#[test]
fn test_convert_config_builder_chain() {
    use nn_models::convert::ConvertConfig;
    let cfg = ConvertConfig::new("test")
        .with_validate_weights(false)
        .with_constant_fold(true);
    assert_eq!(cfg.model_name, "test");
    assert!(!cfg.validate_weights);
    assert!(cfg.constant_fold);
}

#[test]
fn test_convert_config_default_unnamed() {
    use nn_models::convert::ConvertConfig;
    let cfg = ConvertConfig::default();
    assert_eq!(cfg.model_name, "unnamed");
    assert!(cfg.validate_weights, "default should validate weights");
    assert!(cfg.constant_fold, "default should constant fold");
}

// ============================================================================
// ConvertError display
// ============================================================================

#[test]
fn test_convert_error_weight_shape_mismatch_display() {
    use nn_models::convert::ConvertError;
    let err = ConvertError::WeightShapeMismatch {
        name: "encoder.weight".into(),
        expected: 768,
        actual: 512,
    };
    let msg = err.to_string();
    assert!(msg.contains("encoder.weight"), "should name weight: {msg}");
    assert!(msg.contains("768"), "should show expected: {msg}");
    assert!(msg.contains("512"), "should show actual: {msg}");
}

#[test]
fn test_convert_error_graph_parse_display() {
    use nn_models::convert::ConvertError;
    let err = ConvertError::GraphParse("unexpected token".into());
    let msg = err.to_string();
    assert!(
        msg.contains("unexpected token"),
        "should contain detail: {msg}"
    );
}

// ============================================================================
// TransformerBuildError display
// ============================================================================

#[test]
fn test_transformer_build_error_weight_size_display() {
    let err = TransformerBuildError::WeightSize {
        name: "attn.q_weight".into(),
        expected: 1024,
        actual: 512,
    };
    let msg = err.to_string();
    assert!(msg.contains("attn.q_weight"), "should name weight: {msg}");
    assert!(msg.contains("1024"), "should show expected: {msg}");
    assert!(msg.contains("512"), "should show actual: {msg}");
}

#[test]
fn test_transformer_build_error_dim_mismatch_display() {
    let err = TransformerBuildError::DimMismatch {
        stage: "encoder layer 3".into(),
        expected: 768,
        actual: 256,
    };
    let msg = err.to_string();
    assert!(msg.contains("encoder layer 3"), "should name stage: {msg}");
}

// ============================================================================
// DemucsBuilderError display
// ============================================================================

#[test]
fn test_demucs_builder_error_weight_size_display() {
    let err = DemucsBuilderError::WeightSize {
        name: "conv.weight".into(),
        expected: 4096,
        actual: 100,
    };
    let msg = err.to_string();
    assert!(msg.contains("conv.weight"), "should name weight: {msg}");
}

#[test]
fn test_demucs_builder_error_invalid_conv_dim_display() {
    let err = DemucsBuilderError::InvalidConvDim {
        msg: "zero stride in encoder block 2".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("zero stride"), "should contain message: {msg}");
}

#[test]
fn test_demucs_builder_error_block_count_mismatch_display() {
    let err = DemucsBuilderError::BlockCountMismatch {
        context: "spectral_encoder".into(),
        expected: 4,
        actual: 3,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("spectral_encoder"),
        "should name context: {msg}"
    );
    assert!(msg.contains("4"), "should show expected: {msg}");
    assert!(msg.contains("3"), "should show actual: {msg}");
}

// ============================================================================
// Demucs shared constants
// ============================================================================

#[test]
fn test_demucs_base_channels() {
    assert_eq!(BASE_CHANNELS, 48, "HTDemucs base channels = 48");
}

#[test]
fn test_demucs_audio_channels() {
    assert_eq!(AUDIO_CHANNELS, 2, "stereo audio = 2 channels");
}

#[test]
fn test_demucs_channels_at_depth_formula() {
    // channels_at_depth(d) = BASE_CHANNELS * GROWTH^d
    for d in 0..4 {
        let expected = (BASE_CHANNELS as f64 * GROWTH.powi(d as i32)) as usize;
        assert_eq!(
            channels_at_depth(d),
            expected,
            "channels_at_depth({d}) = {expected}"
        );
    }
}

#[test]
fn test_demucs_spectral_depth() {
    // Full HTDemucs: 6 spectral depths (4 basic + 2 deep with BiLSTM + attention)
    assert_eq!(SPECTRAL_DEPTH, 6, "HTDemucs has 6 spectral encoder levels");
    assert_eq!(
        SPECTRAL_BASIC_DEPTH, 4,
        "HTDemucs has 4 basic spectral blocks (depths 0-3)"
    );
}

#[test]
fn test_demucs_conv1d_output_len_basic() {
    // output_len = (input_len + 2*padding - kernel_size) / stride + 1
    let out = conv1d_output_len(100, 3, 1, 1).unwrap();
    assert_eq!(out, 100, "k=3, s=1, p=1 preserves length");
}
