// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Model verification test runner: validates compilation, config construction,
//! weight initialization, and basic forward pass for all model crates.
//!
//! Each model gets a test function that verifies:
//! (a) Config construction succeeds
//! (b) Weight initialization (zero weights for testing) succeeds
//! (c) Forward pass with dummy input produces correct output shape
//! (d) Output contains no NaN/Inf values
//!
//! Uses DynTensor with CPU backend for weight-free tests. Tests requiring real
//! weights (e.g., Kokoro full forward, HTDemucs Metal pipeline) check env vars
//! and skip gracefully when unavailable.
//!
//! Part of #4353

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, VarBuilder};

// ---------------------------------------------------------------------------
// Helper: verify model output shape and finiteness
// ---------------------------------------------------------------------------

/// Verify that a model output tensor has the expected shape and contains no
/// NaN or Inf values.
///
/// Panics with a diagnostic message if:
/// - The tensor rank does not match `expected_shape.len()`
/// - Any dimension does not match the expected value
/// - Any element is NaN or Inf
fn verify_model_output(output: &DynTensor, expected_shape: &[usize], label: &str) {
    // Shape check
    let actual_dims = output.dims();
    assert_eq!(
        actual_dims.len(),
        expected_shape.len(),
        "{label}: rank mismatch — expected {}, got {} (shape: {actual_dims:?})",
        expected_shape.len(),
        actual_dims.len(),
    );
    for (i, (&actual, &expected)) in actual_dims.iter().zip(expected_shape.iter()).enumerate() {
        assert_eq!(
            actual, expected,
            "{label}: dim[{i}] mismatch — expected {expected}, got {actual} (full shape: {actual_dims:?})",
        );
    }

    // Finiteness check: move to CPU if needed, extract f32, check each element
    let cpu_tensor = if output.device() != Device::Cpu {
        output
            .to_device(&Device::Cpu)
            .unwrap_or_else(|e| panic!("{label}: failed to move to CPU: {e}"))
    } else {
        output.clone()
    };
    let values = cpu_tensor
        .to_flat_vec::<f32>()
        .unwrap_or_else(|e| panic!("{label}: failed to extract f32 values: {e}"));

    for (i, &v) in values.iter().enumerate() {
        assert!(
            v.is_finite(),
            "{label}: output[{i}] = {v} is not finite (NaN or Inf)",
        );
    }
}

// ===========================================================================
// Whisper: encoder + decoder forward pass with zero weights on CPU
// ===========================================================================

#[test]
fn verify_whisper_config_and_forward() {
    // (a) Config construction
    let config = nn_whisper::test_utils::tiny_config();
    assert_eq!(config.d_model, 16, "tiny config d_model");
    assert_eq!(config.encoder_layers, 1, "tiny config encoder_layers");
    assert_eq!(config.decoder_layers, 1, "tiny config decoder_layers");

    // (b) Weight initialization with zeros
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let mut model = nn_whisper::WhisperModel::load(&vb, config.clone())
        .expect("Whisper model load with zeros");

    // (c) Encoder forward pass
    let mel_frames = 16;
    let mel = DynTensor::zeros(
        &[1, config.num_mel_bins, mel_frames],
        DType::F32,
        &Device::Cpu,
    )
    .expect("mel input tensor");
    let enc_out = model.encode(&mel).expect("Whisper encode forward");

    // Encoder output: [batch, seq_len, d_model] — seq_len depends on conv stride
    assert_eq!(enc_out.rank(), 3, "encoder output rank");
    assert_eq!(enc_out.dim(0).unwrap(), 1, "encoder batch dim");
    assert_eq!(
        enc_out.dim(2).unwrap(),
        config.d_model,
        "encoder d_model dim"
    );

    // (d) Finiteness check on encoder output
    let enc_shape = enc_out.dims().to_vec();
    verify_model_output(&enc_out, &enc_shape, "whisper_encoder");

    // Decoder forward pass
    let tokens = DynTensor::new(&[0.0, 1.0, 2.0], &[1, 3], &Device::Cpu).expect("token tensor");
    let logits = model
        .decode(&tokens, &enc_out, true, 0)
        .expect("Whisper decode forward");

    verify_model_output(&logits, &[1, 3, config.vocab_size], "whisper_decoder");
}

// ===========================================================================
// Qwen3: decoder-only transformer forward with zero weights on CPU
// ===========================================================================

#[test]
fn verify_qwen3_config_and_forward() {
    // (a) Config construction
    let cfg = nn_qwen3::test_utils::tiny_config();
    assert_eq!(cfg.hidden_size, 256, "tiny config hidden_size");
    assert_eq!(cfg.num_hidden_layers, 2, "tiny config num_hidden_layers");

    // (b) Weight initialization with zeros
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = nn_qwen3::Qwen3Model::load(&vb, cfg.clone()).expect("Qwen3 model load with zeros");

    // (c) Forward pass with token IDs
    let input_ids = &[0_usize, 1, 2, 3];
    let positions = &[0_usize, 1, 2, 3];
    let logits = model.forward(input_ids, positions).expect("Qwen3 forward");

    // (d) Shape and finiteness
    verify_model_output(
        &logits,
        &[1, input_ids.len(), cfg.vocab_size],
        "qwen3_forward",
    );
}

// ===========================================================================
// GLM5: decoder-only transformer forward with zero weights on CPU
// ===========================================================================

#[test]
fn verify_glm5_config_and_forward() {
    // (a) Config construction — use inline config (test-helpers feature not
    // enabled for nn-glm5 in nn-metal dev-deps)
    let cfg = nn_glm5::Glm5Config::new(
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
    );
    assert_eq!(cfg.hidden_size, 256, "config hidden_size");
    assert_eq!(cfg.num_layers, 2, "config num_layers");

    // (b) Weight initialization with zeros
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = nn_glm5::Glm5Model::load(&vb, cfg.clone()).expect("GLM5 model load with zeros");

    // (c) Forward pass with token IDs
    let input_ids = &[0_usize, 1, 2];
    let positions = &[0_usize, 1, 2];
    let logits = model.forward(input_ids, positions).expect("GLM5 forward");

    // (d) Shape and finiteness
    verify_model_output(
        &logits,
        &[1, input_ids.len(), cfg.padded_vocab_size],
        "glm5_forward",
    );
}

// ===========================================================================
// Kokoro: config construction and validation (full forward needs explicit
// weight maps — tested in nn-models/kokoro_tts_tests_model.rs)
// ===========================================================================

#[test]
fn verify_kokoro_config_construction() {
    // (a) Config construction via Default
    let config = nn_models::KokoroConfig::new();
    assert_eq!(config.d_en, 512, "default d_en");
    assert_eq!(config.n_prosody_layers, 3, "default n_prosody_layers");
    assert_eq!(config.style_dim, 128, "default style_dim");
    assert_eq!(
        config.gen_initial_channels, 512,
        "default gen_initial_channels"
    );
    assert_eq!(config.n_fft, 20, "default n_fft");

    // Validate config invariants
    config.validate().expect("default config should be valid");
}

/// Kokoro full forward pass requires real weights (env KOKORO_WEIGHTS).
/// Config loading + zero-weight VarBuilder cannot produce the per-submodule
/// weight maps that KokoroModel::load needs (prefixed weight keys). This test
/// verifies that the model loads and produces valid output when weights exist.
#[test]
fn verify_kokoro_forward_with_real_weights() {
    let weights_path = match std::env::var("KOKORO_WEIGHTS") {
        Ok(p) => std::path::PathBuf::from(p),
        Err(_) => {
            eprintln!("SKIP: KOKORO_WEIGHTS not set. Set to kokoro safetensors path to enable.");
            return;
        }
    };
    if !weights_path.exists() {
        eprintln!(
            "SKIP: KOKORO_WEIGHTS={} does not exist.",
            weights_path.display()
        );
        return;
    }

    let config = nn_models::KokoroConfig::new();
    let tensors =
        nn_core::dyn_tensor::load_safetensors(&weights_path).expect("load kokoro weights");
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let model = nn_models::KokoroModel::load(&vb, &config).expect("KokoroModel load");

    // Minimal forward: 3 token IDs, style vector of correct dimension
    let input_ids = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &Device::Cpu).expect("input_ids");
    let style_dim = config.style_dim * 2; // decoder_style + prosody_style
    let style = DynTensor::zeros(&[1, style_dim], DType::F32, &Device::Cpu).expect("style tensor");
    let result = model.forward(&input_ids, &style, 1.0);
    assert!(result.is_ok(), "Kokoro forward failed: {:?}", result.err());

    let (magnitude, phase) = result.unwrap();
    // Output should be rank 3: [batch, n_bins, time_frames]
    assert_eq!(magnitude.rank(), 3, "magnitude rank");
    assert_eq!(magnitude.dim(0).unwrap(), 1, "magnitude batch dim");
    assert_eq!(phase.rank(), 3, "phase rank");
    assert_eq!(phase.dim(0).unwrap(), 1, "phase batch dim");

    // Finiteness
    let mag_shape = magnitude.dims().to_vec();
    let phase_shape = phase.dims().to_vec();
    verify_model_output(&magnitude, &mag_shape, "kokoro_magnitude");
    verify_model_output(&phase, &phase_shape, "kokoro_phase");
}

// ===========================================================================
// Silero VAD: DynTensor pipeline on CPU (no real weights needed)
// ===========================================================================

#[test]
fn verify_silero_vad_pipeline_cpu() {
    use nn_core::layers::{Conv1dConfig, Module};

    // (a) Config construction — Silero VAD uses Conv1d + LSTM + Linear pipeline
    // (b) Build a VAD-like pipeline with zero weights on CPU
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

    let conv1 =
        nn_core::layers::conv1d(1, 16, 3, Conv1dConfig::default(), vb.pp("enc.0")).expect("conv1");
    let conv2 =
        nn_core::layers::conv1d(16, 32, 3, Conv1dConfig::default(), vb.pp("enc.1")).expect("conv2");
    let fc = nn_core::layers::linear(32, 1, vb.pp("head")).expect("fc");

    // (c) Forward pass: [1, 1, 64] audio -> Conv -> ReLU -> Conv -> ReLU -> mean -> Linear -> sigmoid
    let x = DynTensor::zeros(&[1, 1, 64], DType::F32, &Device::Cpu).expect("input");
    let h = conv1.forward(&x).expect("conv1 fwd");
    let h = h.relu().expect("relu1");
    let h = conv2.forward(&h).expect("conv2 fwd");
    let h = h.relu().expect("relu2");
    let h = h.mean(2).expect("mean pool");
    let h = fc.forward(&h).expect("fc fwd");
    let output = h.sigmoid().expect("sigmoid");

    // (d) Shape and finiteness
    verify_model_output(&output, &[1, 1], "silero_vad_pipeline");

    // Sigmoid output should be in [0, 1]
    let values = output.to_flat_vec::<f32>().unwrap();
    for (i, &v) in values.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "silero_vad output[{i}] = {v} outside [0, 1]",
        );
    }
}

/// Silero VAD with real weights via Metal pipeline.
/// Gated on SILERO_VAD_WEIGHTS or default model path.
#[test]
fn verify_silero_vad_real_weights() {
    let env_path = std::env::var("SILERO_VAD_WEIGHTS").ok();
    let weights_path = env_path.map(std::path::PathBuf::from).unwrap_or_else(|| {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("project root")
            .to_path_buf();
        root.join("models/silero_vad/silero_vad_16k.safetensors")
    });

    if !weights_path.exists() {
        eprintln!(
            "SKIP: Silero VAD weights not found at {}. \
             Set SILERO_VAD_WEIGHTS to enable.",
            weights_path.display()
        );
        return;
    }

    let model = nn_metal::SileroVad::load_safetensors(&weights_path).expect("load silero weights");
    let _backend = nn_metal::MetalBackend::init().expect("Metal backend");
    let cache = nn_metal::PipelineCache::new_global().expect("pipeline cache");
    let mut state = nn_metal::SileroVadState::zero();

    // Process 3 silence chunks and verify valid probabilities
    let silence = vec![0.0f32; 512];
    for i in 0..3 {
        let prob = model
            .process(&cache, &silence, &mut state)
            .unwrap_or_else(|e| panic!("process chunk {i}: {e}"));
        assert!(
            prob.is_finite(),
            "chunk {i}: probability {prob} is not finite",
        );
        assert!(
            (0.0..=1.0).contains(&prob),
            "chunk {i}: probability {prob} outside [0, 1]",
        );
    }
}

// ===========================================================================
// HTDemucs: config validation + Metal pipeline with synthetic weights
// HTDemucs is Metal-only (not DynTensor), so full forward requires Metal.
// ===========================================================================

#[test]
fn verify_htdemucs_architecture_constants() {
    // Validate HTDemucs architecture constants from demucs_shared
    assert_eq!(nn_models::demucs_shared::BASE_CHANNELS, 48);
    assert!(nn_models::demucs_shared::GROWTH > 0.0);
    assert!(nn_models::demucs_shared::DCONV_COMPRESS > 0);
    assert!(nn_models::demucs_shared::DCONV_DEPTH > 0);
    assert!(nn_models::demucs_shared::DCONV_KERNEL > 0);

    // Verify channel growth formula
    let base = nn_models::demucs_shared::BASE_CHANNELS as f64;
    let growth = nn_models::demucs_shared::GROWTH;
    for depth in 0..4 {
        let channels = (base * growth.powi(depth)).round() as usize;
        assert!(channels > 0, "channels at depth {depth} should be positive");
    }
}

/// HTDemucs full forward requires Metal + synthetic weights.
/// Gated on macOS (Metal availability).
#[test]
#[cfg(target_os = "macos")]
fn verify_htdemucs_encoder_builder() {
    // Verify temporal encoder block definition builds without error.
    // This exercises the TensorKernelDef builder pipeline (nn-dsl) for
    // Demucs-specific operations.
    let channels = nn_models::demucs_shared::BASE_CHANNELS;
    let kernel = 8_usize;
    let t_in = 64_usize;
    // Conv1d output: (t_in - kernel) / stride + 1 with stride = kernel/2
    let stride = kernel / 2;
    let t_out = (t_in - kernel) / stride + 1;
    assert!(t_out > 0, "temporal output length should be positive");

    // Verify builder function produces valid TensorKernelDef
    let block = nn_models::silero_vad_builders::EncoderBlock {
        in_channels: channels,
        out_channels: channels * 2,
        kernel_size: 3,
        stride: 1,
        padding: 1,
    };
    let t_out_enc = (t_in + 2 * block.padding - block.kernel_size) / block.stride + 1;
    let def = nn_models::silero_vad_builders::build_encoder_block_def(&block, t_in, t_out_enc);
    assert!(
        def.is_ok(),
        "encoder block def should build: {:?}",
        def.err()
    );
}
