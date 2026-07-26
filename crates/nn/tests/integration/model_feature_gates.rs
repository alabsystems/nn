// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests proving model feature re-export paths work.
//!
//! Each feature (`whisper`, `qwen3`, `glm5`) must be importable via the
//! top-level `nn` crate — dvoice and other consumers should never depend
//! on model crates directly.
//!
//! Run all:
//!   cargo test -p nn --test model_feature_gates --features whisper,qwen3,glm5
//!
//! Run per-feature:
//!   cargo test -p nn --test model_feature_gates --features whisper
//!   cargo test -p nn --test model_feature_gates --features qwen3
//!   cargo test -p nn --test model_feature_gates --features glm5

// ---------------------------------------------------------------------------
// Whisper
// ---------------------------------------------------------------------------

#[cfg(feature = "whisper")]
mod whisper_tests {
    use nn::{DType, Device, VarBuilder};

    fn cpu() -> Device {
        Device::Cpu
    }

    fn tiny_whisper_config() -> nn::whisper::WhisperConfig {
        // WhisperConfig is #[non_exhaustive] — must use Default + field mutation.
        let mut cfg = nn::whisper::WhisperConfig::default();
        cfg.num_mel_bins = 4;
        cfg.max_source_positions = 8;
        cfg.d_model = 16;
        cfg.encoder_attention_heads = 2;
        cfg.encoder_layers = 1;
        cfg.encoder_ffn_dim = 32;
        cfg.vocab_size = 32;
        cfg.max_target_positions = 16;
        cfg.decoder_attention_heads = 2;
        cfg.decoder_layers = 1;
        cfg.decoder_ffn_dim = 32;
        cfg
    }

    #[test]
    fn test_whisper_reexport_path() {
        // Verify consumer import path: `nn::whisper::WhisperConfig`
        let cfg = tiny_whisper_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_whisper_model_loads_via_nn() {
        let cfg = tiny_whisper_config();
        let vb = VarBuilder::zeros(DType::F32, &cpu());
        let model = nn::whisper::WhisperModel::load(&vb, cfg);
        assert!(
            model.is_ok(),
            "WhisperModel::load via nn::whisper failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_whisper_error_type_accessible() {
        // Consumer should be able to match on error types
        let err = nn::whisper::WhisperError::InvalidConfig {
            reason: "test".into(),
        };
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_whisper_decode_types_accessible() {
        // Key decode types available via nn::whisper
        let _config = nn::whisper::DecodeConfig::default();
    }

    #[test]
    fn test_whisper_audio_constants() {
        // Audio processing constants must be accessible
        assert_eq!(nn::whisper::SAMPLE_RATE, 16_000);
        assert_eq!(nn::whisper::HOP_LENGTH, 160);
    }
}

// ---------------------------------------------------------------------------
// Qwen3
// ---------------------------------------------------------------------------

#[cfg(feature = "qwen3")]
mod qwen3_tests {
    use nn::{DType, Device, VarBuilder};

    fn cpu() -> Device {
        Device::Cpu
    }

    fn tiny_qwen3_config() -> nn::Qwen3Config {
        nn::Qwen3Config::new(
            256,      // hidden_size: num_heads * head_dim = 2 * 128
            512,      // intermediate_size
            2,        // num_hidden_layers
            2,        // num_attention_heads
            2,        // num_key_value_heads
            100,      // vocab_size
            1e-6,     // rms_norm_eps
            10_000.0, // rope_theta
            64,       // max_position_embeddings
            true,     // tie_word_embeddings
            None,     // rope_scaling
        )
    }

    #[test]
    fn test_qwen3_config_reexport_path() {
        // Verify consumer import: `nn::Qwen3Config`
        let cfg = tiny_qwen3_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_qwen3_model_loads_via_nn() {
        let cfg = tiny_qwen3_config();
        let vb = VarBuilder::zeros(DType::F32, &cpu());
        let model = nn::Qwen3Model::load(&vb, cfg);
        assert!(
            model.is_ok(),
            "Qwen3Model::load via nn:: failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_qwen3_error_type_accessible() {
        let err = nn::Qwen3Error::InvalidConfig {
            reason: "test".into(),
        };
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_qwen3_forward_tiny() {
        let cfg = tiny_qwen3_config();
        let vb = VarBuilder::zeros(DType::F32, &cpu());
        let model = nn::Qwen3Model::load(&vb, cfg).unwrap();

        // Single-token forward pass: input_ids and positions as &[usize]
        let input_ids: &[usize] = &[1];
        let positions: &[usize] = &[0];
        let result = model.forward(input_ids, positions);
        assert!(result.is_ok(), "Qwen3 forward failed: {:?}", result.err());

        let output = result.unwrap();
        assert_eq!(output.dims()[0], 1); // batch_size
        assert_eq!(output.dims()[1], 1); // seq_len (single token)
        assert_eq!(output.dims()[2], 100); // vocab_size
    }
}

// ---------------------------------------------------------------------------
// GLM-5
// ---------------------------------------------------------------------------

#[cfg(feature = "glm5")]
mod glm5_tests {
    use nn::{DType, Device, VarBuilder};

    fn cpu() -> Device {
        Device::Cpu
    }

    fn tiny_glm5_config() -> nn::Glm5Config {
        nn::Glm5Config::new(
            256,      // hidden_size
            512,      // ffn_hidden_size
            2,        // num_layers
            4,        // num_attention_heads
            2,        // multi_query_group_num
            100,      // padded_vocab_size
            64,       // kv_channels
            1e-5,     // layernorm_epsilon
            64,       // seq_length
            true,     // rmsnorm
            true,     // add_qkv_bias
            false,    // add_bias_linear
            10_000.0, // rope_theta
        )
    }

    #[test]
    fn test_glm5_config_reexport_path() {
        // Verify consumer import: `nn::Glm5Config`
        let cfg = tiny_glm5_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_glm5_model_loads_via_nn() {
        let cfg = tiny_glm5_config();
        let vb = VarBuilder::zeros(DType::F32, &cpu());
        let model = nn::Glm5Model::load(&vb, cfg);
        assert!(
            model.is_ok(),
            "Glm5Model::load via nn:: failed: {:?}",
            model.err()
        );
    }

    #[test]
    fn test_glm5_error_type_accessible() {
        let err = nn::Glm5Error::InvalidConfig {
            reason: "test".into(),
        };
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_glm5_forward_tiny() {
        let cfg = tiny_glm5_config();
        let vb = VarBuilder::zeros(DType::F32, &cpu());
        let model = nn::Glm5Model::load(&vb, cfg).unwrap();

        // Single-token forward pass: input_ids and positions as &[usize]
        let input_ids: &[usize] = &[1];
        let positions: &[usize] = &[0];
        let result = model.forward(input_ids, positions);
        assert!(result.is_ok(), "GLM-5 forward failed: {:?}", result.err());

        let output = result.unwrap();
        assert_eq!(output.dims()[0], 1); // batch_size
        assert_eq!(output.dims()[1], 1); // seq_len (single token)
        assert_eq!(output.dims()[2], 100); // padded_vocab_size
    }
}
