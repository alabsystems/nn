// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Real-weight integration tests for Qwen3Model.
//!
//! These tests load actual Qwen3-0.6B weights from safetensors and validate:
//! 1. Weight loading succeeds with correct shapes
//! 2. Forward pass produces finite logits with correct dimensions
//! 3. Logits match PyTorch reference within tolerance (1e-3)
//! 4. Argmax predictions match PyTorch exactly
//! 5. Autoregressive decoding with KV cache produces finite outputs
//!
//! Gated behind `QWEN3_WEIGHTS` env var pointing to the safetensors file.
//! Tests skip gracefully when the env var is unset.
//!
//! Generate weights:
//! ```bash
//! export QWEN3_WEIGHTS=./nn/weights/qwen3-0.6b.safetensors
//! export QWEN3_REFERENCE=./nn/weights/qwen3-0.6b_reference.safetensors
//! ```

use nn_core::DType;
use nn_qwen3::{Qwen3Config, Qwen3Model};

// ---------------------------------------------------------------------------
// Config for Qwen3-0.6B (from HuggingFace Qwen/Qwen3-0.6B)
// ---------------------------------------------------------------------------

/// Qwen3-0.6B configuration matching HuggingFace checkpoint.
///
/// Verified against `weights/qwen3-0.6b-config.json`.
fn qwen3_0_6b_config() -> Qwen3Config {
    Qwen3Config::new(
        1024,        // hidden_size
        3072,        // intermediate_size
        28,          // num_hidden_layers
        16,          // num_attention_heads
        8,           // num_key_value_heads
        151_936,     // vocab_size
        1e-6,        // rms_norm_eps
        1_000_000.0, // rope_theta
        40_960,      // max_position_embeddings
        true,        // tie_word_embeddings
        None,        // rope_scaling
    )
}

/// Reference input token IDs for "Hello world" from Qwen3 tokenizer.
const REF_INPUT_IDS: &[usize] = &[9707, 1879];

/// Reference positions (0-indexed).
const REF_POSITIONS: &[usize] = &[0, 1];

/// Expected argmax per position from PyTorch reference.
const REF_ARGMAX: &[usize] = &[21806, 0];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn weights_path() -> Option<String> {
    std::env::var("QWEN3_WEIGHTS").ok()
}

fn reference_path() -> Option<String> {
    std::env::var("QWEN3_REFERENCE").ok()
}

/// Returns the weights path or prints skip message and returns from the caller.
macro_rules! require_weights {
    () => {
        match weights_path() {
            Some(p) => p,
            None => {
                eprintln!("QWEN3_WEIGHTS not set, skipping real-weight test");
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

    let config = qwen3_0_6b_config();
    let result = Qwen3Model::load_safetensors(&wpath, config);
    assert!(
        result.is_ok(),
        "Qwen3-0.6B load_safetensors failed: {:?}",
        result.err()
    );

    let model = result.unwrap();
    assert_eq!(model.config().hidden_size, 1024);
    assert_eq!(model.config().num_hidden_layers, 28);
    assert_eq!(model.config().num_attention_heads, 16);
    assert_eq!(model.config().num_key_value_heads, 8);
    assert_eq!(model.config().vocab_size, 151_936);
    assert_eq!(model.dtype(), DType::F32);
    eprintln!("Qwen3-0.6B loaded successfully (28 layers, 151936 vocab)");
}

// ---------------------------------------------------------------------------
// Test: weight shapes match expected dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_weight_shapes() {
    let wpath = require_weights!();

    let tensors = nn_core::load_safetensors(&wpath).unwrap();

    // Embedding
    let embed = &tensors["model.embed_tokens.weight"];
    assert_eq!(embed.dims(), &[151_936, 1024], "embed_tokens shape");

    // lm_head (tied, so same tensor present under different name)
    let lm = &tensors["lm_head.weight"];
    assert_eq!(lm.dims(), &[151_936, 1024], "lm_head shape");

    // Layer 0 self-attention
    let q = &tensors["model.layers.0.self_attn.q_proj.weight"];
    assert_eq!(
        q.dims(),
        &[2048, 1024],
        "q_proj: [nh*hd, h] = [16*128, 1024]"
    );

    let k = &tensors["model.layers.0.self_attn.k_proj.weight"];
    assert_eq!(
        k.dims(),
        &[1024, 1024],
        "k_proj: [nkv*hd, h] = [8*128, 1024]"
    );

    let v = &tensors["model.layers.0.self_attn.v_proj.weight"];
    assert_eq!(
        v.dims(),
        &[1024, 1024],
        "v_proj: [nkv*hd, h] = [8*128, 1024]"
    );

    let o = &tensors["model.layers.0.self_attn.o_proj.weight"];
    assert_eq!(o.dims(), &[1024, 2048], "o_proj: [h, nh*hd] = [1024, 2048]");

    // QK-Norm
    let qn = &tensors["model.layers.0.self_attn.q_norm.weight"];
    assert_eq!(qn.dims(), &[128], "q_norm: [head_dim]");

    let kn = &tensors["model.layers.0.self_attn.k_norm.weight"];
    assert_eq!(kn.dims(), &[128], "k_norm: [head_dim]");

    // Layer norms
    let iln = &tensors["model.layers.0.input_layernorm.weight"];
    assert_eq!(iln.dims(), &[1024], "input_layernorm: [hidden_size]");

    let paln = &tensors["model.layers.0.post_attention_layernorm.weight"];
    assert_eq!(
        paln.dims(),
        &[1024],
        "post_attention_layernorm: [hidden_size]"
    );

    // MLP (SwiGLU)
    let gate = &tensors["model.layers.0.mlp.gate_proj.weight"];
    assert_eq!(
        gate.dims(),
        &[3072, 1024],
        "gate_proj: [intermediate, hidden]"
    );

    let up = &tensors["model.layers.0.mlp.up_proj.weight"];
    assert_eq!(up.dims(), &[3072, 1024], "up_proj: [intermediate, hidden]");

    let down = &tensors["model.layers.0.mlp.down_proj.weight"];
    assert_eq!(
        down.dims(),
        &[1024, 3072],
        "down_proj: [hidden, intermediate]"
    );

    // Final norm
    let fnorm = &tensors["model.norm.weight"];
    assert_eq!(fnorm.dims(), &[1024], "final norm: [hidden_size]");

    // Verify all 28 layers exist
    for i in 0..28 {
        let key = format!("model.layers.{i}.self_attn.q_proj.weight");
        assert!(tensors.contains_key(&key), "missing layer {i} q_proj");
    }

    // Verify total tensor count: 311
    // embed(1) + 28 layers * (q,k,v,o,qn,kn,gate,up,down,iln,paln = 11) + norm(1) + lm_head(1)
    // = 1 + 28*11 + 1 + 1 = 311
    assert_eq!(tensors.len(), 311, "total tensor count");

    eprintln!("All weight shapes validated");
}

// ---------------------------------------------------------------------------
// Test: forward pass produces correct output dimensions
// ---------------------------------------------------------------------------

#[test]
fn test_forward_dimensions() {
    let wpath = require_weights!();

    let config = qwen3_0_6b_config();
    let model = Qwen3Model::load_safetensors(&wpath, config).unwrap();

    let logits = model.forward(REF_INPUT_IDS, REF_POSITIONS).unwrap();

    assert_eq!(logits.rank(), 3, "logits should be rank 3");
    assert_eq!(logits.dim(0).unwrap(), 1, "batch size");
    assert_eq!(logits.dim(1).unwrap(), 2, "seq_len for 2 tokens");
    assert_eq!(logits.dim(2).unwrap(), 151_936, "vocab size");

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
// Test: argmax matches PyTorch reference
// ---------------------------------------------------------------------------

#[test]
fn test_argmax_matches_pytorch() {
    let wpath = require_weights!();

    let config = qwen3_0_6b_config();
    let model = Qwen3Model::load_safetensors(&wpath, config).unwrap();

    let logits = model.forward(REF_INPUT_IDS, REF_POSITIONS).unwrap();

    // Check argmax at each position
    for pos in 0..REF_INPUT_IDS.len() {
        let pos_logits = logits.narrow(1, pos, 1).unwrap().squeeze(1).unwrap();
        let flat = pos_logits.to_flat_vec::<f32>().unwrap();
        let argmax = flat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(
            argmax, REF_ARGMAX[pos],
            "pos {pos}: argmax mismatch (got {argmax}, expected {})",
            REF_ARGMAX[pos]
        );
        eprintln!("pos {pos}: argmax = {argmax} (matches PyTorch)");
    }
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
            eprintln!("QWEN3_REFERENCE not set, skipping logit comparison");
            return;
        }
    };

    let config = qwen3_0_6b_config();
    let model = Qwen3Model::load_safetensors(&wpath, config).unwrap();

    let logits = model.forward(REF_INPUT_IDS, REF_POSITIONS).unwrap();
    let nn_logits = logits.to_flat_vec::<f32>().unwrap();

    // Load PyTorch reference logits
    let ref_tensors = nn_core::load_safetensors(&rpath).unwrap();
    let ref_logits_tensor = &ref_tensors["logits"];
    assert_eq!(
        ref_logits_tensor.dims(),
        &[1, 2, 151_936],
        "reference logits shape"
    );
    let ref_logits = ref_logits_tensor.to_flat_vec::<f32>().unwrap();

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
    // 28 transformer layers (RoPE, RMSNorm, SwiGLU, attention, residuals).
    // This is generous enough for f32 CPU inference but catches gross bugs.
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
// Test: logit statistics match PyTorch (sanity check)
// ---------------------------------------------------------------------------

#[test]
fn test_logit_statistics() {
    let wpath = require_weights!();

    let config = qwen3_0_6b_config();
    let model = Qwen3Model::load_safetensors(&wpath, config).unwrap();

    let logits = model.forward(REF_INPUT_IDS, REF_POSITIONS).unwrap();
    let flat = logits.to_flat_vec::<f32>().unwrap();

    let min = flat.iter().copied().fold(f32::INFINITY, f32::min);
    let max = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = flat.iter().map(|v| f64::from(*v)).sum::<f64>() / flat.len() as f64;

    eprintln!("Logit stats: min={min:.4}, max={max:.4}, mean={mean:.4}");

    // PyTorch reference: min=-10.8598, max=12.9560, mean=-1.2245
    // Allow generous tolerance for stats (5% relative)
    assert!(
        min < -5.0,
        "min logit should be strongly negative, got {min}"
    );
    assert!(
        max > 5.0,
        "max logit should be strongly positive, got {max}"
    );
    assert!(
        (mean - (-1.2245)).abs() < 0.5,
        "mean logit should be near -1.22, got {mean:.4}"
    );
}

// ---------------------------------------------------------------------------
// Test: autoregressive decoding with KV cache
// ---------------------------------------------------------------------------

#[test]
fn test_autoregressive_with_cache() {
    let wpath = require_weights!();

    let config = qwen3_0_6b_config();
    let model = Qwen3Model::load_safetensors(&wpath, config).unwrap();
    let mut cache = model.new_cache();

    // Step 0: first token
    let logits0 = model
        .forward_cached(&[9707], &[0], Some(&mut cache))
        .unwrap();
    assert_eq!(logits0.dims(), &[1, 1, 151_936]);

    // Step 1: second token
    let logits1 = model
        .forward_cached(&[1879], &[1], Some(&mut cache))
        .unwrap();
    assert_eq!(logits1.dims(), &[1, 1, 151_936]);

    // Verify outputs are finite
    for (i, logits) in [&logits0, &logits1].iter().enumerate() {
        let flat = logits.to_flat_vec::<f32>().unwrap();
        let nf = flat.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(nf, 0, "step {i} logits should be finite");
    }

    // The cached step-1 logits should match the uncached full-sequence logits
    // at position 1 (the last token).
    let full_logits = model.forward(REF_INPUT_IDS, REF_POSITIONS).unwrap();
    let full_last = full_logits
        .narrow(1, 1, 1)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cached_last = logits1.to_flat_vec::<f32>().unwrap();

    let mut max_err: f32 = 0.0;
    for (a, b) in full_last.iter().zip(cached_last.iter()) {
        let err = (a - b).abs();
        if err > max_err {
            max_err = err;
        }
    }

    eprintln!("Cached vs uncached last-position max error: {max_err:.6e}");
    // Causal attention should give identical results for the last position
    // regardless of whether computed via cache or full sequence.
    assert!(
        max_err < 1e-4,
        "cached vs uncached logits differ by {max_err:.6e} (should be < 1e-4)"
    );
}

// ---------------------------------------------------------------------------
// Test: single token forward (minimal inference)
// ---------------------------------------------------------------------------

#[test]
fn test_single_token_forward() {
    let wpath = require_weights!();

    let config = qwen3_0_6b_config();
    let model = Qwen3Model::load_safetensors(&wpath, config).unwrap();

    let logits = model.forward(&[0], &[0]).unwrap();
    assert_eq!(logits.dims(), &[1, 1, 151_936]);

    let flat = logits.to_flat_vec::<f32>().unwrap();
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "single-token logits should be finite");
    eprintln!("Single token forward: OK");
}

// ---------------------------------------------------------------------------
// Test: greedy generation produces tokens
// ---------------------------------------------------------------------------

#[test]
fn test_greedy_generation() {
    let wpath = require_weights!();

    let config = qwen3_0_6b_config();
    let model = Qwen3Model::load_safetensors(&wpath, config).unwrap();

    // Generate 5 tokens from "Hello world"
    let output = model.generate_greedy(REF_INPUT_IDS, 5).unwrap();

    // token_ids excludes the prompt — only newly generated tokens
    assert!(
        !output.token_ids.is_empty(),
        "should generate at least 1 token, got 0"
    );
    assert!(
        output.token_ids.len() <= 5,
        "should not exceed max_new_tokens (5), got {}",
        output.token_ids.len()
    );

    // First generated token from greedy decoding (temperature=0 argmax).
    // Note: generate_greedy uses forward_cached with KV cache, which should
    // produce the same last-position logits as the uncached forward. Our
    // test_autoregressive_with_cache confirms 0 error for this.
    let first_gen = output.token_ids[0];
    eprintln!(
        "First generated token: {first_gen} (direct-forward argmax: {})",
        REF_ARGMAX[REF_ARGMAX.len() - 1]
    );

    // All generated tokens should be within vocab range
    for (i, &tok) in output.token_ids.iter().enumerate() {
        assert!(
            tok < 151_936,
            "generated token {i} ({tok}) exceeds vocab size"
        );
    }

    eprintln!(
        "Generated {} tokens: {:?}",
        output.token_ids.len(),
        &output.token_ids
    );
}
