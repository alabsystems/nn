// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Real-weight integration tests for Glm5Model.
//!
//! These tests load actual ChatGLM-4 weights from safetensors and validate:
//! 1. Weight loading succeeds with correct shapes
//! 2. Forward pass produces finite logits with correct dimensions
//! 3. Logits match PyTorch reference within tolerance (1e-3)
//! 4. Argmax predictions match PyTorch exactly
//! 5. Autoregressive decoding with KV cache produces finite outputs
//!
//! Gated behind `GLM5_WEIGHTS` env var pointing to the safetensors file.
//! Tests skip gracefully when the env var is unset.
//!
//! Supports two model variants:
//! - **Tiny** (`optimum-intel-internal-testing/tiny-random-chatglm4`):
//!   ~19 MB, 6 layers, hidden_size=32 (for CI/development)
//! - **Full** (`THUDM/glm-4-9b-chat`):
//!   ~18 GB, 40 layers, hidden_size=4096 (for accuracy validation)
//!
//! Generate weights:
//! ```bash
//! python3 scripts/download_glm4_weights.py          # tiny (default)
//! python3 scripts/download_glm4_weights.py --full    # full 9B model
//!
//! export GLM5_WEIGHTS=./nn/weights/glm4-tiny.safetensors
//! export GLM5_REFERENCE=./nn/weights/glm4-tiny_reference.safetensors
//! cargo test -p nn-glm5 --test real_weights -- --nocapture
//! ```

use nn_core::DType;
use nn_glm5::{Glm5Config, Glm5Model};

// ---------------------------------------------------------------------------
// Model variant detection and config
// ---------------------------------------------------------------------------

/// Detect model variant from GLM5_CONFIG env var or probe the weight file.
///
/// The download script saves a JSON config alongside weights. If `GLM5_CONFIG`
/// is set, reads it. Otherwise, probes the weight file to detect dimensions.
fn detect_config() -> Glm5Config {
    // If GLM5_CONFIG is set, try to read it
    if let Ok(cfg_path) = std::env::var("GLM5_CONFIG") {
        if let Ok(contents) = std::fs::read_to_string(&cfg_path) {
            if let Ok(parsed) = parse_config_json(&contents) {
                return parsed;
            }
        }
    }

    // Probe the weight file to detect dimensions
    if let Some(wpath) = weights_path() {
        if let Ok(tensors) = nn_core::load_safetensors(&wpath) {
            if let Some(embed) = tensors.get("transformer.embedding.word_embeddings.weight") {
                let dims = embed.dims();
                if dims.len() == 2 {
                    let padded_vocab_size = dims[0];
                    let hidden_size = dims[1];
                    return config_from_hidden_size(hidden_size, padded_vocab_size);
                }
            }
        }
    }

    // Default: GLM-4-9B-chat
    Glm5Config::glm4_9b_chat()
}

/// Parse a config JSON (from download script) into Glm5Config.
fn parse_config_json(contents: &str) -> Result<Glm5Config, String> {
    // Minimal JSON parsing: extract hf_config fields
    // The download script writes: {"hf_config": {"hidden_size": ..., ...}}
    // We use a simple approach since we don't want to add serde_json as a dep
    let find_usize = |key: &str| -> Option<usize> {
        let pattern = format!("\"{key}\": ");
        contents.find(&pattern).and_then(|pos| {
            let start = pos + pattern.len();
            let rest = &contents[start..];
            let end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            rest[..end].parse().ok()
        })
    };

    let find_f64 = |key: &str| -> Option<f64> {
        let pattern = format!("\"{key}\": ");
        contents.find(&pattern).and_then(|pos| {
            let start = pos + pattern.len();
            let rest = &contents[start..];
            let end = rest
                .find(|c: char| {
                    !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+'
                })
                .unwrap_or(rest.len());
            rest[..end].parse().ok()
        })
    };

    let hidden_size = find_usize("hidden_size").ok_or("missing hidden_size")?;
    let ffn_hidden_size = find_usize("ffn_hidden_size").ok_or("missing ffn_hidden_size")?;
    let num_layers = find_usize("num_layers").ok_or("missing num_layers")?;
    let num_attention_heads =
        find_usize("num_attention_heads").ok_or("missing num_attention_heads")?;
    let multi_query_group_num =
        find_usize("multi_query_group_num").ok_or("missing multi_query_group_num")?;
    let padded_vocab_size = find_usize("padded_vocab_size").ok_or("missing padded_vocab_size")?;
    let kv_channels = find_usize("kv_channels").ok_or("missing kv_channels")?;
    let seq_length = find_usize("seq_length").ok_or("missing seq_length")?;
    let layernorm_epsilon = find_f64("layernorm_epsilon").ok_or("missing layernorm_epsilon")?;
    let rope_theta = find_f64("rope_theta").ok_or("missing rope_theta")?;

    Ok(Glm5Config::new(
        hidden_size,
        ffn_hidden_size,
        num_layers,
        num_attention_heads,
        multi_query_group_num,
        padded_vocab_size,
        kv_channels,
        layernorm_epsilon,
        seq_length,
        true,  // rmsnorm
        true,  // add_qkv_bias
        false, // add_bias_linear
        rope_theta,
    ))
}

/// Build config from detected hidden_size. Supports known variants.
fn config_from_hidden_size(hidden_size: usize, padded_vocab_size: usize) -> Glm5Config {
    match hidden_size {
        // Tiny: optimum-intel-internal-testing/tiny-random-chatglm4
        32 => Glm5Config::new(
            32, // hidden_size
            64, // ffn_hidden_size
            6,  // num_layers
            4,  // num_attention_heads
            2,  // multi_query_group_num
            padded_vocab_size,
            8,           // kv_channels
            1.5625e-7,   // layernorm_epsilon
            131_072,     // seq_length
            true,        // rmsnorm
            true,        // add_qkv_bias
            false,       // add_bias_linear
            5_000_000.0, // rope_theta (rope_ratio=500)
        ),
        // Full: THUDM/glm-4-9b-chat
        4096 => Glm5Config::glm4_9b_chat(),
        // Unknown: try to construct a reasonable config
        _ => {
            eprintln!("WARNING: unknown hidden_size={hidden_size}, using GLM-4-9B-chat defaults");
            Glm5Config::glm4_9b_chat()
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn weights_path() -> Option<String> {
    std::env::var("GLM5_WEIGHTS").ok()
}

fn reference_path() -> Option<String> {
    std::env::var("GLM5_REFERENCE").ok()
}

/// Returns the weights path or prints skip message and returns from the caller.
macro_rules! require_weights {
    () => {
        match weights_path() {
            Some(p) => p,
            None => {
                eprintln!("GLM5_WEIGHTS not set, skipping real-weight test");
                return;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Test: weight loading and key validation
// ---------------------------------------------------------------------------

#[test]
fn test_load_real_weights() {
    let wpath = require_weights!();
    let config = detect_config();

    let result = Glm5Model::load_safetensors(&wpath, config.clone());
    assert!(
        result.is_ok(),
        "load_safetensors failed: {:?}",
        result.err()
    );

    let model = result.unwrap();
    assert_eq!(model.config().hidden_size, config.hidden_size);
    assert_eq!(model.config().num_layers, config.num_layers);
    assert_eq!(
        model.config().num_attention_heads,
        config.num_attention_heads
    );
    assert_eq!(
        model.config().multi_query_group_num,
        config.multi_query_group_num
    );
    assert_eq!(model.config().padded_vocab_size, config.padded_vocab_size);
    assert_eq!(model.config().kv_channels, config.kv_channels);
    assert_eq!(model.dtype(), DType::F32);
    eprintln!(
        "Loaded: {} layers, hidden_size={}, vocab={}",
        config.num_layers, config.hidden_size, config.padded_vocab_size
    );
}

// ---------------------------------------------------------------------------
// Test: weight shapes match expected dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_weight_shapes() {
    let wpath = require_weights!();
    let config = detect_config();

    let tensors = nn_core::load_safetensors(&wpath).unwrap();
    let h = config.hidden_size;
    let ffn = config.ffn_hidden_size;
    let nh = config.num_attention_heads;
    let nkv = config.multi_query_group_num;
    let hd = config.kv_channels;
    let vocab = config.padded_vocab_size;

    // Embedding: [padded_vocab_size, hidden_size]
    let embed = &tensors["transformer.embedding.word_embeddings.weight"];
    assert_eq!(embed.dims(), &[vocab, h], "embedding shape");

    // Output layer: [padded_vocab_size, hidden_size]
    let output = &tensors["transformer.output_layer.weight"];
    assert_eq!(output.dims(), &[vocab, h], "output_layer shape");

    // Final layernorm: [hidden_size]
    let fnorm = &tensors["transformer.encoder.final_layernorm.weight"];
    assert_eq!(fnorm.dims(), &[h], "final_layernorm shape");

    // Layer 0 self-attention: fused QKV
    let qkv_size = (nh + 2 * nkv) * hd;
    let qkv_w = &tensors["transformer.encoder.layers.0.self_attention.query_key_value.weight"];
    assert_eq!(
        qkv_w.dims(),
        &[qkv_size, h],
        "qkv weight: [(nh+2*nkv)*hd, h]"
    );

    // QKV bias (add_qkv_bias=true for all ChatGLM models)
    let qkv_b = &tensors["transformer.encoder.layers.0.self_attention.query_key_value.bias"];
    assert_eq!(qkv_b.dims(), &[qkv_size], "qkv bias");

    // Dense (output projection): [hidden_size, num_heads * head_dim]
    let dense = &tensors["transformer.encoder.layers.0.self_attention.dense.weight"];
    assert_eq!(dense.dims(), &[h, nh * hd], "dense weight: [h, nh*hd]");

    // Layer norms: [hidden_size]
    let iln = &tensors["transformer.encoder.layers.0.input_layernorm.weight"];
    assert_eq!(iln.dims(), &[h], "input_layernorm");

    let paln = &tensors["transformer.encoder.layers.0.post_attention_layernorm.weight"];
    assert_eq!(paln.dims(), &[h], "post_attention_layernorm");

    // MLP: dense_h_to_4h [ffn*2, h] (SwiGLU gate+up fused)
    let h_to_4h = &tensors["transformer.encoder.layers.0.mlp.dense_h_to_4h.weight"];
    assert_eq!(h_to_4h.dims(), &[ffn * 2, h], "dense_h_to_4h: [ffn*2, h]");

    // MLP: dense_4h_to_h [h, ffn]
    let h_from_4h = &tensors["transformer.encoder.layers.0.mlp.dense_4h_to_h.weight"];
    assert_eq!(h_from_4h.dims(), &[h, ffn], "dense_4h_to_h: [h, ffn]");

    // Verify all layers exist
    for i in 0..config.num_layers {
        let key = format!("transformer.encoder.layers.{i}.self_attention.query_key_value.weight");
        assert!(tensors.contains_key(&key), "missing layer {i} qkv weight");
    }

    // Expected tensor count: embed(1) + layers * 7 + final_ln(1) + output(1)
    // Per-layer: input_ln, qkv_w, qkv_b, dense, post_ln, h_to_4h, 4h_to_h = 7
    let expected_count = 1 + config.num_layers * 7 + 1 + 1;
    assert_eq!(
        tensors.len(),
        expected_count,
        "total tensor count (expected {expected_count}, got {})",
        tensors.len()
    );

    eprintln!("All weight shapes validated ({} tensors)", tensors.len());
}

// ---------------------------------------------------------------------------
// Test: forward pass produces correct output dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_forward_dimensions() {
    let wpath = require_weights!();
    let config = detect_config();
    let vocab = config.padded_vocab_size;
    let model = Glm5Model::load_safetensors(&wpath, config).unwrap();

    // Single token forward
    let logits = model.forward(&[0], &[0]).unwrap();

    assert_eq!(logits.rank(), 3, "logits should be rank 3");
    assert_eq!(logits.dim(0).unwrap(), 1, "batch size");
    assert_eq!(logits.dim(1).unwrap(), 1, "seq_len for 1 token");
    assert_eq!(logits.dim(2).unwrap(), vocab, "vocab size");

    let flat = logits.to_flat_vec::<f32>().unwrap();
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "all logits should be finite");

    eprintln!(
        "Forward pass: shape [{}, {}, {}], all finite",
        logits.dim(0).unwrap(),
        logits.dim(1).unwrap(),
        logits.dim(2).unwrap()
    );
}

// ---------------------------------------------------------------------------
// Test: multi-token forward pass
// ---------------------------------------------------------------------------

#[test]
fn test_multi_token_forward() {
    let wpath = require_weights!();
    let config = detect_config();
    let vocab = config.padded_vocab_size;
    let model = Glm5Model::load_safetensors(&wpath, config).unwrap();

    // Multi-token forward (triggers causal mask path)
    let input_ids: Vec<usize> = vec![0, 1, 2, 3, 4];
    let positions: Vec<usize> = vec![0, 1, 2, 3, 4];
    let logits = model.forward(&input_ids, &positions).unwrap();

    assert_eq!(logits.dims(), &[1, 5, vocab]);

    let flat = logits.to_flat_vec::<f32>().unwrap();
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "all logits should be finite");

    eprintln!("Multi-token forward: shape {:?}, all finite", logits.dims());
}

// ---------------------------------------------------------------------------
// Test: logits match PyTorch reference within tolerance
// ---------------------------------------------------------------------------

#[test]
fn test_logits_match_pytorch_reference() {
    let wpath = require_weights!();

    let rpath = match reference_path() {
        Some(p) => p,
        None => {
            eprintln!("GLM5_REFERENCE not set, skipping logit comparison");
            return;
        }
    };

    let config = detect_config();
    let model = Glm5Model::load_safetensors(&wpath, config).unwrap();

    // Load reference data
    let ref_tensors = nn_core::load_safetensors(&rpath).unwrap();
    let ref_input_ids_tensor = &ref_tensors["input_ids"];
    let ref_logits_tensor = &ref_tensors["logits"];

    // Extract input IDs from reference (stored as float32)
    let ref_input_ids_flat = ref_input_ids_tensor.to_flat_vec::<f32>().unwrap();
    let input_ids: Vec<usize> = ref_input_ids_flat.iter().map(|v| *v as usize).collect();
    let positions: Vec<usize> = (0..input_ids.len()).collect();

    eprintln!(
        "Reference input IDs: {:?} (seq_len={})",
        &input_ids,
        input_ids.len()
    );

    // Forward pass
    let logits = model.forward(&input_ids, &positions).unwrap();
    let nn_logits = logits.to_flat_vec::<f32>().unwrap();
    let ref_logits = ref_logits_tensor.to_flat_vec::<f32>().unwrap();

    assert_eq!(
        logits.dims(),
        ref_logits_tensor.dims(),
        "logits shape mismatch: nn={:?}, pytorch={:?}",
        logits.dims(),
        ref_logits_tensor.dims()
    );
    assert_eq!(nn_logits.len(), ref_logits.len(), "logit vector lengths");

    // Compute max absolute error and mean absolute error
    let mut max_abs_err: f32 = 0.0;
    let mut sum_abs_err: f64 = 0.0;
    let mut max_err_idx = 0;
    for (i, (a, b)) in nn_logits.iter().zip(ref_logits.iter()).enumerate() {
        let err = (a - b).abs();
        sum_abs_err += f64::from(err);
        if err > max_abs_err {
            max_abs_err = err;
            max_err_idx = i;
        }
    }
    let mean_abs_err = sum_abs_err / nn_logits.len() as f64;

    eprintln!("Logit comparison ({} values):", nn_logits.len());
    eprintln!("  max absolute error: {max_abs_err:.6e} at index {max_err_idx}");
    eprintln!("  mean absolute error: {mean_abs_err:.6e}");

    // Tolerance: 1e-3 for accumulated floating point differences across
    // transformer layers (half-RoPE, RMSNorm, SwiGLU, attention, residuals).
    // Generous enough for f32 CPU inference but catches gross bugs.
    let tolerance = 1e-3;
    assert!(
        max_abs_err < tolerance,
        "max absolute error {max_abs_err:.6e} exceeds tolerance {tolerance:.0e} \
         at index {max_err_idx} (nn={:.6}, pytorch={:.6})",
        nn_logits[max_err_idx],
        ref_logits[max_err_idx]
    );

    eprintln!("All logits within tolerance {tolerance:.0e}");
}

// ---------------------------------------------------------------------------
// Test: argmax matches PyTorch reference
// ---------------------------------------------------------------------------

#[test]
fn test_argmax_matches_pytorch() {
    let wpath = require_weights!();

    let rpath = match reference_path() {
        Some(p) => p,
        None => {
            eprintln!("GLM5_REFERENCE not set, skipping argmax comparison");
            return;
        }
    };

    let config = detect_config();
    let vocab = config.padded_vocab_size;
    let model = Glm5Model::load_safetensors(&wpath, config).unwrap();

    // Load reference
    let ref_tensors = nn_core::load_safetensors(&rpath).unwrap();
    let ref_input_ids_tensor = &ref_tensors["input_ids"];
    let ref_logits_tensor = &ref_tensors["logits"];

    let ref_input_ids_flat = ref_input_ids_tensor.to_flat_vec::<f32>().unwrap();
    let input_ids: Vec<usize> = ref_input_ids_flat.iter().map(|v| *v as usize).collect();
    let positions: Vec<usize> = (0..input_ids.len()).collect();

    let logits = model.forward(&input_ids, &positions).unwrap();
    let nn_logits = logits.to_flat_vec::<f32>().unwrap();
    let ref_logits = ref_logits_tensor.to_flat_vec::<f32>().unwrap();

    let seq_len = input_ids.len();

    for pos in 0..seq_len {
        let nn_slice = &nn_logits[pos * vocab..(pos + 1) * vocab];
        let nn_argmax = nn_slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let ref_slice = &ref_logits[pos * vocab..(pos + 1) * vocab];
        let ref_argmax = ref_slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(
            nn_argmax, ref_argmax,
            "pos {pos}: argmax mismatch (nn={nn_argmax}, pytorch={ref_argmax})"
        );
        eprintln!("pos {pos}: argmax = {nn_argmax} (matches PyTorch)");
    }
}

// ---------------------------------------------------------------------------
// Test: logit statistics are reasonable
// ---------------------------------------------------------------------------

#[test]
fn test_logit_statistics() {
    let wpath = require_weights!();
    let config = detect_config();
    let model = Glm5Model::load_safetensors(&wpath, config).unwrap();

    let logits = model.forward(&[0, 1, 2], &[0, 1, 2]).unwrap();
    let flat = logits.to_flat_vec::<f32>().unwrap();

    let min = flat.iter().copied().fold(f32::INFINITY, f32::min);
    let max = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = flat.iter().map(|v| f64::from(*v)).sum::<f64>() / flat.len() as f64;

    eprintln!("Logit stats: min={min:.4}, max={max:.4}, mean={mean:.4}");

    // All logits must be finite
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "all logits should be finite");

    // Range should be non-trivial for production models. The tiny random
    // model (hidden_size=32) has all-zero embeddings, producing zero logits.
    // Accept zero range only for the tiny model.
    let range = max - min;
    let config = detect_config();
    if config.hidden_size > 64 {
        assert!(
            range > 0.01,
            "logit range should be non-trivial for production models, got {range:.6}"
        );
    } else {
        eprintln!("  (tiny model: accepting zero-range logits from all-zero embeddings)");
    }
}

// ---------------------------------------------------------------------------
// Test: autoregressive decoding with KV cache
// ---------------------------------------------------------------------------

#[test]
fn test_autoregressive_with_cache() {
    let wpath = require_weights!();
    let config = detect_config();
    let vocab = config.padded_vocab_size;
    let model = Glm5Model::load_safetensors(&wpath, config).unwrap();
    let mut cache = model.new_cache();

    // Step 0: first token
    let logits0 = model.forward_cached(&[0], &[0], Some(&mut cache)).unwrap();
    assert_eq!(logits0.dims(), &[1, 1, vocab]);

    // Step 1: second token
    let logits1 = model.forward_cached(&[1], &[1], Some(&mut cache)).unwrap();
    assert_eq!(logits1.dims(), &[1, 1, vocab]);

    // Step 2: third token
    let logits2 = model.forward_cached(&[2], &[2], Some(&mut cache)).unwrap();
    assert_eq!(logits2.dims(), &[1, 1, vocab]);

    // All outputs should be finite
    for (i, logits) in [&logits0, &logits1, &logits2].iter().enumerate() {
        let flat = logits.to_flat_vec::<f32>().unwrap();
        let nf = flat.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(nf, 0, "step {i} logits should be finite");
    }

    eprintln!("Autoregressive cache: 3 steps, all finite");
}

// ---------------------------------------------------------------------------
// Test: cached vs uncached consistency
// ---------------------------------------------------------------------------

#[test]
fn test_cached_vs_uncached_consistency() {
    let wpath = require_weights!();
    let config = detect_config();
    let model = Glm5Model::load_safetensors(&wpath, config).unwrap();
    let mut cache = model.new_cache();

    // Cached: token-by-token
    let _logits0 = model.forward_cached(&[0], &[0], Some(&mut cache)).unwrap();
    let logits1_cached = model.forward_cached(&[1], &[1], Some(&mut cache)).unwrap();

    // Uncached: full sequence
    let full_logits = model.forward(&[0, 1], &[0, 1]).unwrap();
    let full_last = full_logits
        .narrow(1, 1, 1)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cached_last = logits1_cached.to_flat_vec::<f32>().unwrap();

    let mut max_err: f32 = 0.0;
    for (a, b) in full_last.iter().zip(cached_last.iter()) {
        let err = (a - b).abs();
        if err > max_err {
            max_err = err;
        }
    }

    eprintln!("Cached vs uncached last-position max error: {max_err:.6e}");
    // Causal attention should give identical results for the last position.
    assert!(
        max_err < 1e-4,
        "cached vs uncached logits differ by {max_err:.6e} (should be < 1e-4)"
    );
}

// ---------------------------------------------------------------------------
// Test: second reference input (multi-token, from download script)
// ---------------------------------------------------------------------------

#[test]
fn test_logits_match_pytorch_reference_input2() {
    let wpath = require_weights!();

    let rpath = match reference_path() {
        Some(p) => p,
        None => {
            eprintln!("GLM5_REFERENCE not set, skipping input2 comparison");
            return;
        }
    };

    let config = detect_config();
    let model = Glm5Model::load_safetensors(&wpath, config).unwrap();

    let ref_tensors = nn_core::load_safetensors(&rpath).unwrap();

    // Check if input2 exists in reference
    if !ref_tensors.contains_key("input_ids2") {
        eprintln!("Reference file missing input_ids2, skipping");
        return;
    }

    let ref_input_ids2 = &ref_tensors["input_ids2"];
    let ref_logits2 = &ref_tensors["logits2"];

    let input_ids: Vec<usize> = ref_input_ids2
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| *v as usize)
        .collect();
    let positions: Vec<usize> = (0..input_ids.len()).collect();

    eprintln!("Input2 IDs: {:?}", &input_ids);

    let logits = model.forward(&input_ids, &positions).unwrap();
    let nn_logits = logits.to_flat_vec::<f32>().unwrap();
    let pytorch_logits = ref_logits2.to_flat_vec::<f32>().unwrap();

    assert_eq!(
        nn_logits.len(),
        pytorch_logits.len(),
        "logit vector lengths"
    );

    let mut max_abs_err: f32 = 0.0;
    for (a, b) in nn_logits.iter().zip(pytorch_logits.iter()) {
        let err = (a - b).abs();
        if err > max_abs_err {
            max_abs_err = err;
        }
    }

    eprintln!("Input2 max absolute error: {max_abs_err:.6e}");

    let tolerance = 1e-3;
    assert!(
        max_abs_err < tolerance,
        "input2 max absolute error {max_abs_err:.6e} exceeds tolerance {tolerance:.0e}"
    );
}
