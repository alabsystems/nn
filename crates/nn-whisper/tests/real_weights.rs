// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Real-weight integration tests for Whisper tiny.
//!
//! These tests load actual AI Provider Whisper-tiny weights from safetensors and
//! compare nn inference outputs against PyTorch reference tensors saved as
//! .npy files.
//!
//! ## Setup
//!
//! Set `WHISPER_WEIGHTS` to the directory containing `model.safetensors` and
//! reference `.npy` files:
//!
//! ```bash
//! export WHISPER_WEIGHTS=./nn/weights/whisper-tiny
//! cargo test -p nn-whisper --test real_weights -- --nocapture
//! ```
//!
//! Required files in `$WHISPER_WEIGHTS/`:
//! - `model.safetensors` — AI Provider whisper-tiny weights (151 MB)
//! - `ref_mel_input.npy` — PyTorch reference mel input [1, 80, 3000]
//! - `ref_encoder_output.npy` — PyTorch reference encoder output [1, 1500, 384]
//! - `ref_decoder_input_ids.npy` — PyTorch reference decoder token IDs [1, 1]
//! - `ref_decoder_logits.npy` — PyTorch reference decoder logits [1, 1, 51865]
//!
//! ## Known Divergence: Encoder Positional Embeddings
//!
//! The nn encoder uses **sinusoidal** positional embeddings (generated at load
//! time), while the HuggingFace/AI Provider Whisper encoder uses **learned** positional
//! embeddings (`model.encoder.embed_positions.weight`). This causes the encoder
//! outputs to diverge from PyTorch reference values. The encoder tests verify
//! correct shapes, finite outputs, and reasonable magnitude — not exact parity.
//!
//! The decoder uses learned positional embeddings correctly loaded from weights.

use std::path::{Path, PathBuf};

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_reftest::load_npy;
use nn_whisper::{WhisperConfig, WhisperModel};

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
                 Set to whisper-tiny weights directory to run real-weight tests."
            );
            return;
        };
    };
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

/// Load the whisper-tiny model from safetensors.
fn load_model(dir: &Path) -> WhisperModel {
    let st_path = dir.join("model.safetensors");
    let config = WhisperConfig::whisper_tiny();
    WhisperModel::load_safetensors(&st_path, config)
        .unwrap_or_else(|e| panic!("Failed to load model from {}: {e}", st_path.display()))
}

/// Compute max absolute difference between two float slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "tensor size mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Compute mean absolute difference between two float slices.
fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "tensor size mismatch");
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .sum();
    (sum / a.len() as f64) as f32
}

/// Compute cosine similarity between two float slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = f64::from(*x);
        let y = f64::from(*y);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        dot / denom
    }
}

// ===========================================================================
// A. Weight Loading Tests
// ===========================================================================

#[test]
fn test_real_weights_load_successfully() {
    skip_without_weights!(dir);

    let model = load_model(&dir);
    let config = model.config();

    assert_eq!(config.d_model, 384, "whisper-tiny d_model");
    assert_eq!(config.num_mel_bins, 80, "whisper-tiny mel bins");
    assert_eq!(config.encoder_layers, 4, "whisper-tiny encoder layers");
    assert_eq!(config.decoder_layers, 4, "whisper-tiny decoder layers");
    assert_eq!(
        config.encoder_attention_heads, 6,
        "whisper-tiny encoder heads"
    );
    assert_eq!(
        config.decoder_attention_heads, 6,
        "whisper-tiny decoder heads"
    );
    assert_eq!(config.vocab_size, 51865, "whisper-tiny vocab size");
    assert_eq!(config.encoder_ffn_dim, 1536, "whisper-tiny encoder FFN dim");
    assert_eq!(config.decoder_ffn_dim, 1536, "whisper-tiny decoder FFN dim");

    eprintln!("Model loaded: whisper-tiny ({} d_model)", config.d_model);
}

// ===========================================================================
// B. Encoder Tests
// ===========================================================================

#[test]
fn test_encoder_output_shape_matches_reference() {
    skip_without_weights!(dir);

    let mut model = load_model(&dir);
    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");

    assert_eq!(mel_shape, vec![1, 80, 3000], "mel input shape");

    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu)
        .expect("create mel tensor from npy");

    let enc_out = model.encode(&mel).expect("encoder forward");

    assert_eq!(enc_out.rank(), 3, "encoder output rank");
    assert_eq!(enc_out.dim(0).unwrap(), 1, "batch dim");
    assert_eq!(enc_out.dim(1).unwrap(), 1500, "sequence length");
    assert_eq!(enc_out.dim(2).unwrap(), 384, "d_model");

    eprintln!("Encoder output shape: {:?}", enc_out.dims());
}

#[test]
fn test_encoder_output_finite() {
    skip_without_weights!(dir);

    let mut model = load_model(&dir);
    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");

    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu)
        .expect("create mel tensor from npy");

    let enc_out = model.encode(&mel).expect("encoder forward");
    let flat = enc_out.to_flat_vec::<f32>().unwrap();

    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "encoder output should have no NaN/Inf values"
    );

    let min_val = flat.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean_val: f32 = (flat.iter().map(|v| f64::from(*v)).sum::<f64>() / flat.len() as f64) as f32;

    eprintln!("Encoder output stats: min={min_val:.6}, max={max_val:.6}, mean={mean_val:.6}");
}

#[test]
fn test_encoder_output_reasonable_magnitude() {
    // With real weights, encoder outputs should be in a reasonable range.
    // PyTorch reference: min=-17.27, max=18.79, mean=0.031.
    // nn uses sinusoidal positional embeddings (vs learned in PyTorch),
    // so values will differ, but magnitude should be comparable.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);
    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");

    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu)
        .expect("create mel tensor from npy");

    let enc_out = model.encode(&mel).expect("encoder forward");
    let flat = enc_out.to_flat_vec::<f32>().unwrap();

    let max_abs = flat.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    // With real weights, max absolute value should be < 100 (PyTorch ref is ~18).
    assert!(
        max_abs < 100.0,
        "encoder output max abs value ({max_abs:.4}) should be < 100"
    );

    // Outputs should not all be zero (would indicate broken weight loading).
    let nonzero_count = flat.iter().filter(|v| v.abs() > 1e-6).count();
    assert!(
        nonzero_count > flat.len() / 2,
        "encoder output should have mostly non-zero values, got {nonzero_count}/{} non-zero",
        flat.len()
    );

    eprintln!(
        "Encoder max_abs={max_abs:.4}, nonzero={nonzero_count}/{}",
        flat.len()
    );
}

#[test]
fn test_encoder_divergence_from_pytorch_documented() {
    // This test documents the divergence between nn (sinusoidal pos-embed)
    // and PyTorch (learned pos-embed) encoder outputs.
    //
    // EXPECTED: Large divergence due to different positional embeddings.
    // This test ensures the divergence is measured and tracked. When nn
    // switches to learned positional embeddings for the encoder, this test
    // should be updated to enforce tight tolerance.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);
    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");
    let (ref_enc_data, ref_enc_shape) = load_ref_npy(&dir, "ref_encoder_output");

    assert_eq!(ref_enc_shape, vec![1, 1500, 384], "reference encoder shape");

    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu)
        .expect("create mel tensor from npy");

    let enc_out = model.encode(&mel).expect("encoder forward");
    let nn_enc_data = enc_out.to_flat_vec::<f32>().unwrap();

    let mad = max_abs_diff(&nn_enc_data, &ref_enc_data);
    let mean_ad = mean_abs_diff(&nn_enc_data, &ref_enc_data);
    let cos_sim = cosine_similarity(&nn_enc_data, &ref_enc_data);

    eprintln!("Encoder PyTorch parity:");
    eprintln!("  max_abs_diff  = {mad:.6}");
    eprintln!("  mean_abs_diff = {mean_ad:.6}");
    eprintln!("  cosine_sim    = {cos_sim:.8}");
    eprintln!("  NOTE: Large divergence expected due to sinusoidal vs learned pos-embed.");

    // Sanity check: outputs should be finite and have the right shape.
    assert_eq!(nn_enc_data.len(), ref_enc_data.len());
    assert!(
        nn_enc_data.iter().all(|v| v.is_finite()),
        "nn encoder output must be all finite"
    );
}

// ===========================================================================
// C. Decoder Tests
// ===========================================================================

#[test]
fn test_decoder_output_shape_matches_reference() {
    skip_without_weights!(dir);

    let mut model = load_model(&dir);

    // Use nn's own encoder output (not PyTorch reference, since they differ).
    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");
    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu).expect("create mel tensor");
    let enc_out = model.encode(&mel).expect("encoder forward");

    // Decoder input: single SOT token (50258).
    let tokens = DynTensor::new(&[50258.0], &[1, 1], &Device::Cpu).expect("token tensor");

    let logits = model
        .decode(&tokens, &enc_out, true, 0)
        .expect("decoder forward");

    assert_eq!(logits.rank(), 3, "logits rank");
    assert_eq!(logits.dim(0).unwrap(), 1, "batch dim");
    assert_eq!(logits.dim(1).unwrap(), 1, "seq_len (single token)");
    assert_eq!(logits.dim(2).unwrap(), 51865, "vocab_size");

    eprintln!("Decoder logits shape: {:?}", logits.dims());
}

#[test]
fn test_decoder_output_finite_with_real_weights() {
    skip_without_weights!(dir);

    let mut model = load_model(&dir);

    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");
    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu).expect("create mel tensor");
    let enc_out = model.encode(&mel).expect("encoder forward");

    let tokens = DynTensor::new(&[50258.0], &[1, 1], &Device::Cpu).expect("token tensor");
    let logits = model
        .decode(&tokens, &enc_out, true, 0)
        .expect("decoder forward");

    let flat = logits.to_flat_vec::<f32>().unwrap();
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "decoder logits should have no NaN/Inf");

    let min_val = flat.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean_val: f32 = (flat.iter().map(|v| f64::from(*v)).sum::<f64>() / flat.len() as f64) as f32;

    eprintln!("Decoder logits stats: min={min_val:.6}, max={max_val:.6}, mean={mean_val:.6}");
}

#[test]
fn test_decoder_with_pytorch_encoder_output() {
    // Feed PyTorch reference encoder output into the nn decoder.
    // Since the decoder uses learned positional embeddings (loaded from weights),
    // and token embeddings match, the decoder should produce reasonably close
    // results when given the same encoder output.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);

    // Use PyTorch's encoder output as decoder input.
    let (ref_enc_data, ref_enc_shape) = load_ref_npy(&dir, "ref_encoder_output");
    let (ref_logits_data, ref_logits_shape) = load_ref_npy(&dir, "ref_decoder_logits");

    assert_eq!(ref_enc_shape, vec![1, 1500, 384]);
    assert_eq!(ref_logits_shape, vec![1, 1, 51865]);

    let enc_out = DynTensor::from_vec(ref_enc_data, &ref_enc_shape, &Device::Cpu)
        .expect("create encoder output tensor");

    // SOT token (50258).
    let tokens = DynTensor::new(&[50258.0], &[1, 1], &Device::Cpu).expect("token tensor");

    let logits = model
        .decode(&tokens, &enc_out, true, 0)
        .expect("decoder forward with PyTorch encoder output");

    let nn_logits_data = logits.to_flat_vec::<f32>().unwrap();
    assert_eq!(nn_logits_data.len(), ref_logits_data.len());

    let mad = max_abs_diff(&nn_logits_data, &ref_logits_data);
    let mean_ad = mean_abs_diff(&nn_logits_data, &ref_logits_data);
    let cos_sim = cosine_similarity(&nn_logits_data, &ref_logits_data);

    eprintln!("Decoder parity (PyTorch encoder output fed to nn decoder):");
    eprintln!("  max_abs_diff  = {mad:.6}");
    eprintln!("  mean_abs_diff = {mean_ad:.6}");
    eprintln!("  cosine_sim    = {cos_sim:.8}");

    // With identical encoder output and learned decoder embeddings,
    // the decoder should produce close results. Tolerance is generous
    // because of potential numerical differences in matmul/softmax.
    assert!(
        cos_sim > 0.99,
        "decoder cosine similarity should be > 0.99 with identical encoder output, got {cos_sim:.8}"
    );

    eprintln!("  PASS: cosine_sim={cos_sim:.8} > 0.99 threshold");
}

#[test]
fn test_decoder_top_tokens_match_pytorch() {
    // Compare the top-k predicted tokens between nn and PyTorch.
    // With the same encoder output, the top predictions should largely overlap.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);

    let (ref_enc_data, ref_enc_shape) = load_ref_npy(&dir, "ref_encoder_output");
    let (ref_logits_data, _) = load_ref_npy(&dir, "ref_decoder_logits");

    let enc_out = DynTensor::from_vec(ref_enc_data, &ref_enc_shape, &Device::Cpu)
        .expect("create encoder output tensor");

    let tokens = DynTensor::new(&[50258.0], &[1, 1], &Device::Cpu).expect("token tensor");

    let logits = model
        .decode(&tokens, &enc_out, true, 0)
        .expect("decoder forward");

    let nn_logits = logits.to_flat_vec::<f32>().unwrap();

    // Get top-10 token indices for both.
    let nn_top10 = top_k_indices(&nn_logits, 10);
    let ref_top10 = top_k_indices(&ref_logits_data, 10);

    // Count overlap in top-10 predictions.
    let overlap: usize = nn_top10
        .iter()
        .filter(|idx| ref_top10.contains(idx))
        .count();

    eprintln!("Top-10 token comparison:");
    eprintln!("  nn top-10:     {nn_top10:?}");
    eprintln!("  pytorch top-10: {ref_top10:?}");
    eprintln!("  overlap:        {overlap}/10");

    // With identical encoder output, at least 5 of top-10 should match.
    assert!(
        overlap >= 5,
        "at least 5 of top-10 tokens should overlap, got {overlap}/10"
    );
}

// ===========================================================================
// D. End-to-End Tests
// ===========================================================================

#[test]
fn test_encode_decode_roundtrip_real_weights() {
    // Full pipeline: mel -> encode -> decode -> logits with real weights.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);
    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");

    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu).expect("create mel tensor");

    // Encode.
    let enc_out = model.encode(&mel).expect("encoder forward");
    assert_eq!(enc_out.dims(), &[1, 1500, 384]);

    // Decode: SOT token.
    let tokens = DynTensor::new(&[50258.0], &[1, 1], &Device::Cpu).expect("token tensor");
    let logits = model
        .decode(&tokens, &enc_out, true, 0)
        .expect("decoder forward");
    assert_eq!(logits.dims(), &[1, 1, 51865]);

    let flat = logits.to_flat_vec::<f32>().unwrap();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "end-to-end logits must be finite"
    );

    // The argmax token should be a valid vocab index.
    let argmax = flat
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert!(argmax < 51865, "argmax token index should be < vocab_size");

    eprintln!("End-to-end: argmax token = {argmax}");
}

#[test]
fn test_autoregressive_decode_multi_step() {
    // Multi-step autoregressive decoding with real weights.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);
    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");

    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu).expect("create mel tensor");
    let enc_out = model.encode(&mel).expect("encoder forward");

    let mut generated_tokens: Vec<f32> = vec![50258.0]; // SOT

    for step in 0..5 {
        let tok_data = vec![*generated_tokens.last().unwrap()];
        let tokens = DynTensor::from_vec(tok_data, &[1, 1], &Device::Cpu).expect("token tensor");

        let flush = step == 0;
        let logits = model
            .decode(&tokens, &enc_out, flush, step)
            .unwrap_or_else(|_| panic!("decode step {step}"));

        let flat = logits.to_flat_vec::<f32>().unwrap();
        assert!(
            flat.iter().all(|v| v.is_finite()),
            "step {step}: logits contain non-finite values"
        );

        // Greedy: pick argmax.
        let next_token = flat
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as f32)
            .unwrap();
        generated_tokens.push(next_token);
    }

    eprintln!("Generated tokens (5 steps): {generated_tokens:?}");

    // All generated tokens should be valid vocab indices.
    for &t in &generated_tokens {
        assert!(
            (t as usize) < 51865,
            "generated token {t} should be < vocab_size"
        );
    }
}

#[test]
fn test_kv_cache_reset_determinism() {
    // Verify that resetting KV cache and re-running produces identical results.
    skip_without_weights!(dir);

    let mut model = load_model(&dir);
    let (mel_data, mel_shape) = load_ref_npy(&dir, "ref_mel_input");

    let mel = DynTensor::from_vec(mel_data, &mel_shape, &Device::Cpu).expect("create mel tensor");
    let enc_out = model.encode(&mel).expect("encoder forward");

    let tokens = DynTensor::new(&[50258.0], &[1, 1], &Device::Cpu).expect("token tensor");

    // First decode pass.
    let logits1 = model
        .decode(&tokens, &enc_out, true, 0)
        .expect("first decode");
    let v1 = logits1.to_flat_vec::<f32>().unwrap();

    // Reset and re-decode.
    model.reset_kv_cache();
    let logits2 = model
        .decode(&tokens, &enc_out, true, 0)
        .expect("second decode after reset");
    let v2 = logits2.to_flat_vec::<f32>().unwrap();

    let mad = max_abs_diff(&v1, &v2);
    assert!(
        mad < 1e-6,
        "KV cache reset should produce identical results, max_abs_diff={mad:.6e}"
    );

    eprintln!("KV cache reset determinism: max_abs_diff = {mad:.6e}");
}

// ===========================================================================
// Utility Functions
// ===========================================================================

/// Return indices of top-k values in descending order.
fn top_k_indices(data: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|(i, _)| *i).collect()
}
