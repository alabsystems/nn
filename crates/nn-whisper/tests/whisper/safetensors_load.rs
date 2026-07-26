// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `WhisperModel::load_safetensors()`.
//!
//! Constructs synthetic safetensors files with all required Whisper weight keys,
//! loads the model, and runs encoder + decoder forward passes. Validates the
//! full weight-loading pipeline end-to-end without requiring real model weights.

use std::collections::HashMap;

use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::DType;
use nn_whisper::test_utils::tiny_config;
use nn_whisper::{WhisperConfig, WhisperModel};

// ---------------------------------------------------------------------------
// Synthetic weight builders
// ---------------------------------------------------------------------------

type WeightEntry = (String, Vec<usize>, Vec<f32>);

/// Deterministic xorshift64 pseudo-random f32 generator.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 as f64 / u64::MAX as f64) * 0.02 - 0.01) as f32
    }

    fn tensor(&mut self, shape: &[usize]) -> Vec<f32> {
        let n: usize = shape.iter().product();
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Add self-attention projection weights for one block.
fn push_self_attn(out: &mut Vec<WeightEntry>, prefix: &str, d: usize, rng: &mut Rng) {
    out.push((
        format!("{prefix}.q_proj.weight"),
        vec![d, d],
        rng.tensor(&[d, d]),
    ));
    out.push((format!("{prefix}.q_proj.bias"), vec![d], rng.tensor(&[d])));
    out.push((
        format!("{prefix}.k_proj.weight"),
        vec![d, d],
        rng.tensor(&[d, d]),
    ));
    // Note: no k_proj.bias in Whisper.
    out.push((
        format!("{prefix}.v_proj.weight"),
        vec![d, d],
        rng.tensor(&[d, d]),
    ));
    out.push((format!("{prefix}.v_proj.bias"), vec![d], rng.tensor(&[d])));
    out.push((
        format!("{prefix}.out_proj.weight"),
        vec![d, d],
        rng.tensor(&[d, d]),
    ));
    out.push((format!("{prefix}.out_proj.bias"), vec![d], rng.tensor(&[d])));
}

/// Add FFN weights (layer norm + fc1 + fc2) for one block.
fn push_ffn(out: &mut Vec<WeightEntry>, bp: &str, d: usize, ffn: usize, rng: &mut Rng) {
    out.push((
        format!("{bp}.final_layer_norm.weight"),
        vec![d],
        vec![1.0; d],
    ));
    out.push((format!("{bp}.final_layer_norm.bias"), vec![d], vec![0.0; d]));
    out.push((
        format!("{bp}.fc1.weight"),
        vec![ffn, d],
        rng.tensor(&[ffn, d]),
    ));
    out.push((format!("{bp}.fc1.bias"), vec![ffn], rng.tensor(&[ffn])));
    out.push((
        format!("{bp}.fc2.weight"),
        vec![d, ffn],
        rng.tensor(&[d, ffn]),
    ));
    out.push((format!("{bp}.fc2.bias"), vec![d], rng.tensor(&[d])));
}

/// Build all encoder weights under `model.encoder.*`.
fn build_encoder_weights(config: &WhisperConfig, rng: &mut Rng) -> Vec<WeightEntry> {
    let d = config.d_model;
    let mel = config.num_mel_bins;
    let ffn = config.encoder_ffn_dim;
    let prefix = "model.encoder";
    let mut w = Vec::new();

    // Conv1d stem.
    w.push((
        format!("{prefix}.conv1.weight"),
        vec![d, mel, 3],
        rng.tensor(&[d, mel, 3]),
    ));
    w.push((format!("{prefix}.conv1.bias"), vec![d], rng.tensor(&[d])));
    w.push((
        format!("{prefix}.conv2.weight"),
        vec![d, d, 3],
        rng.tensor(&[d, d, 3]),
    ));
    w.push((format!("{prefix}.conv2.bias"), vec![d], rng.tensor(&[d])));

    // Transformer blocks.
    for i in 0..config.encoder_layers {
        let bp = format!("{prefix}.layers.{i}");
        w.push((
            format!("{bp}.self_attn_layer_norm.weight"),
            vec![d],
            vec![1.0; d],
        ));
        w.push((
            format!("{bp}.self_attn_layer_norm.bias"),
            vec![d],
            vec![0.0; d],
        ));
        push_self_attn(&mut w, &format!("{bp}.self_attn"), d, rng);
        push_ffn(&mut w, &bp, d, ffn, rng);
    }

    // Final layer norm.
    w.push((format!("{prefix}.layer_norm.weight"), vec![d], vec![1.0; d]));
    w.push((format!("{prefix}.layer_norm.bias"), vec![d], vec![0.0; d]));
    w
}

/// Build all decoder weights under `model.decoder.*`.
fn build_decoder_weights(config: &WhisperConfig, rng: &mut Rng) -> Vec<WeightEntry> {
    let d = config.d_model;
    let dec_ffn = config.decoder_ffn_dim;
    let prefix = "model.decoder";
    let mut w = Vec::new();

    // Embeddings.
    w.push((
        format!("{prefix}.embed_tokens.weight"),
        vec![config.vocab_size, d],
        rng.tensor(&[config.vocab_size, d]),
    ));
    w.push((
        format!("{prefix}.embed_positions.weight"),
        vec![config.max_target_positions, d],
        rng.tensor(&[config.max_target_positions, d]),
    ));

    // Transformer blocks (self-attn + cross-attn + FFN).
    for i in 0..config.decoder_layers {
        let bp = format!("{prefix}.layers.{i}");
        w.push((
            format!("{bp}.self_attn_layer_norm.weight"),
            vec![d],
            vec![1.0; d],
        ));
        w.push((
            format!("{bp}.self_attn_layer_norm.bias"),
            vec![d],
            vec![0.0; d],
        ));
        push_self_attn(&mut w, &format!("{bp}.self_attn"), d, rng);
        // Cross-attention.
        w.push((
            format!("{bp}.encoder_attn_layer_norm.weight"),
            vec![d],
            vec![1.0; d],
        ));
        w.push((
            format!("{bp}.encoder_attn_layer_norm.bias"),
            vec![d],
            vec![0.0; d],
        ));
        push_self_attn(&mut w, &format!("{bp}.encoder_attn"), d, rng);
        push_ffn(&mut w, &bp, d, dec_ffn, rng);
    }

    // Final layer norm.
    w.push((format!("{prefix}.layer_norm.weight"), vec![d], vec![1.0; d]));
    w.push((format!("{prefix}.layer_norm.bias"), vec![d], vec![0.0; d]));
    w
}

/// Build all required weight tensors for a Whisper model.
fn build_synthetic_weights(config: &WhisperConfig) -> Vec<WeightEntry> {
    let mut rng = Rng::new(42);
    let mut weights = build_encoder_weights(config, &mut rng);
    weights.extend(build_decoder_weights(config, &mut rng));
    weights
}

// ---------------------------------------------------------------------------
// Safetensors file I/O helpers
// ---------------------------------------------------------------------------

/// Write synthetic weights to a safetensors file.
fn write_safetensors(path: &std::path::Path, weights: &[WeightEntry]) {
    let mut data_map: Vec<(String, safetensors::tensor::TensorView<'_>)> = Vec::new();
    for (name, shape, values) in weights {
        let bytes: &[u8] = unsafe {
            // SAFETY: f32 slice to u8 slice — same memory, valid alignment.
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4)
        };
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                .expect("valid tensor view");
        data_map.push((name.clone(), view));
    }
    let metadata: Option<HashMap<String, String>> = None;
    let data = safetensors::tensor::serialize(
        data_map.iter().map(|(n, v)| (n.as_str(), v.clone())),
        metadata,
    )
    .expect("serialize");
    std::fs::write(path, data).expect("write file");
}

/// Create a temp directory for test artifacts.
fn test_dir(suffix: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "nn_whisper_integration_{suffix}_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_load_safetensors_full_model() {
    let config = tiny_config();
    let weights = build_synthetic_weights(&config);
    let dir = test_dir("full_model");
    let path = dir.join("model.safetensors");
    write_safetensors(&path, &weights);

    let result = WhisperModel::load_safetensors(&path, config);
    assert!(
        result.is_ok(),
        "load_safetensors should succeed: {:?}",
        result.err()
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_encoder_forward_with_synthetic_weights() {
    let config = tiny_config();
    let weights = build_synthetic_weights(&config);
    let dir = test_dir("enc_fwd");
    let path = dir.join("model.safetensors");
    write_safetensors(&path, &weights);

    let mut model = WhisperModel::load_safetensors(&path, config.clone()).unwrap();
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let enc_out = model.encode(&mel).expect("encoder forward should succeed");

    assert_eq!(enc_out.rank(), 3);
    assert_eq!(enc_out.dim(0).unwrap(), 1);
    assert_eq!(enc_out.dim(2).unwrap(), config.d_model);

    let flat = enc_out.to_flat_vec::<f32>().unwrap();
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "encoder output should have no NaN/Inf");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_decoder_forward_with_synthetic_weights() {
    let config = tiny_config();
    let weights = build_synthetic_weights(&config);
    let dir = test_dir("dec_fwd");
    let path = dir.join("model.safetensors");
    write_safetensors(&path, &weights);

    let mut model = WhisperModel::load_safetensors(&path, config.clone()).unwrap();
    let enc_out = DynTensor::zeros(&[1, 8, config.d_model], DType::F32, &cpu()).unwrap();
    let tokens = DynTensor::new(&[0.0, 1.0, 2.0], &[1, 3], &cpu()).unwrap();

    let logits = model
        .decode(&tokens, &enc_out, true, 0)
        .expect("decoder forward should succeed");

    assert_eq!(logits.rank(), 3);
    assert_eq!(logits.dim(0).unwrap(), 1);
    assert_eq!(logits.dim(1).unwrap(), 3);
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_encode_then_decode_roundtrip() {
    let config = tiny_config();
    let weights = build_synthetic_weights(&config);
    let dir = test_dir("roundtrip");
    let path = dir.join("model.safetensors");
    write_safetensors(&path, &weights);

    let mut model = WhisperModel::load_safetensors(&path, config.clone()).unwrap();
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let enc_out = model.encode(&mel).unwrap();

    let tokens = DynTensor::new(&[0.0, 1.0], &[1, 2], &cpu()).unwrap();
    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();

    assert_eq!(logits.dim(1).unwrap(), 2);
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size);
}

#[test]
fn test_autoregressive_decode_with_synthetic_weights() {
    let config = tiny_config();
    let weights = build_synthetic_weights(&config);
    let dir = test_dir("autoreg");
    let path = dir.join("model.safetensors");
    write_safetensors(&path, &weights);

    let mut model = WhisperModel::load_safetensors(&path, config.clone()).unwrap();
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let enc_out = model.encode(&mel).unwrap();

    let t0 = DynTensor::new(&[0.0], &[1, 1], &cpu()).unwrap();
    let logits0 = model.decode(&t0, &enc_out, true, 0).unwrap();
    assert_eq!(logits0.dims(), &[1, 1, config.vocab_size]);

    let t1 = DynTensor::new(&[1.0], &[1, 1], &cpu()).unwrap();
    let logits1 = model.decode(&t1, &enc_out, false, 1).unwrap();
    assert_eq!(logits1.dims(), &[1, 1, config.vocab_size]);

    let t2 = DynTensor::new(&[2.0], &[1, 1], &cpu()).unwrap();
    let logits2 = model.decode(&t2, &enc_out, false, 2).unwrap();
    assert_eq!(logits2.dims(), &[1, 1, config.vocab_size]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_missing_weight_key_returns_error() {
    let config = tiny_config();
    let mut weights = build_synthetic_weights(&config);
    weights.retain(|(name, _, _)| name != "model.encoder.conv1.weight");

    let dir = test_dir("missing_key");
    let path = dir.join("model.safetensors");
    write_safetensors(&path, &weights);

    let result = WhisperModel::load_safetensors(&path, config);
    let err = match result {
        Ok(_) => panic!("should fail with missing weight key"),
        Err(e) => e,
    };
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("conv1.weight")
            || err_msg.contains("not found")
            || err_msg.contains("TensorNotFound"),
        "error should reference the missing key, got: {err_msg}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_wrong_shape_weight_returns_error() {
    let config = tiny_config();
    let mut weights = build_synthetic_weights(&config);

    for (name, shape, data) in &mut weights {
        if name == "model.encoder.conv1.weight" {
            let wrong_n = config.d_model * config.d_model * 3;
            *shape = vec![config.d_model, config.d_model, 3];
            *data = vec![0.01; wrong_n];
            break;
        }
    }

    let dir = test_dir("wrong_shape");
    let path = dir.join("model.safetensors");
    write_safetensors(&path, &weights);

    let result = WhisperModel::load_safetensors(&path, config);
    assert!(result.is_err(), "should fail with wrong shape");

    std::fs::remove_dir_all(&dir).ok();
}
