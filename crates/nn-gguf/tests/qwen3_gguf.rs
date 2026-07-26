// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the GGUF → Qwen3 loading pipeline.
//!
//! Tests cover:
//! - Synthetic GGUF binary construction with qwen2 metadata
//! - GGUF parsing → Qwen3GgufConfig extraction
//! - Tensor name mapping (GGUF → HuggingFace convention)
//! - Dequantized tensor loading into DynTensor
//! - Full end-to-end: GGUF → VarBuilder → Qwen3Model → forward pass
//! - Tied embedding handling (lm_head = embed_tokens when output.weight absent)
//! - Optional real-model loading gated behind QWEN3_GGUF_PATH env var

use std::io::Cursor;

use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};
use nn_dsl as _;
use nn_gguf::{gguf_to_hf_name, load_qwen3_tensors, GgufFile, Qwen3GgufConfig};
use nn_qwen3::Qwen3Config;

// ---------------------------------------------------------------------------
// Synthetic GGUF builder helpers
// ---------------------------------------------------------------------------

/// GGUF v3 magic: "GGUF" in little-endian.
const GGUF_MAGIC: u32 = 0x4647_5547;

/// GGUF metadata type IDs.
const META_TYPE_U32: u32 = 4;
const META_TYPE_F32: u32 = 6;
const META_TYPE_STRING: u32 = 8;

/// GGUF tensor dtype for F32.
const GGUF_DTYPE_F32: u32 = 0;

/// Tiny Qwen3 model dimensions that satisfy head_dim() == 128.
/// hidden_size / num_attention_heads = 256 / 2 = 128.
struct TinyQwen3 {
    hidden: usize,
    intermediate: usize,
    heads: usize,
    kv_heads: usize,
    vocab: usize,
    layers: usize,
}

impl TinyQwen3 {
    fn new() -> Self {
        Self {
            hidden: 256,
            intermediate: 512,
            heads: 2,
            kv_heads: 2,
            vocab: 64,
            layers: 1,
        }
    }

    /// head_dim = hidden / heads = 128
    fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
}

/// Append a GGUF string metadata entry to the buffer.
fn meta_string(buf: &mut Vec<u8>, key: &str, val: &str) {
    // Key: length-prefixed string
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    // Type: STRING
    buf.extend_from_slice(&META_TYPE_STRING.to_le_bytes());
    // Value: length-prefixed string
    buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
    buf.extend_from_slice(val.as_bytes());
}

/// Append a GGUF u32 metadata entry to the buffer.
fn meta_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&META_TYPE_U32.to_le_bytes());
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Append a GGUF f32 metadata entry to the buffer.
fn meta_f32(buf: &mut Vec<u8>, key: &str, val: f32) {
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&META_TYPE_F32.to_le_bytes());
    buf.extend_from_slice(&val.to_le_bytes());
}

/// Append a GGUF tensor info entry (name, dims, dtype=F32, byte offset).
fn tensor_info(buf: &mut Vec<u8>, name: &str, dims: &[u64], offset: u64) {
    // Name: length-prefixed string
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    // Number of dimensions
    buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    // Each dimension
    for &d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    // Dtype: F32
    buf.extend_from_slice(&GGUF_DTYPE_F32.to_le_bytes());
    // Offset within tensor data section
    buf.extend_from_slice(&offset.to_le_bytes());
}

/// Compute byte size of an F32 tensor given its dimensions.
fn f32_byte_size(dims: &[u64]) -> u64 {
    dims.iter().product::<u64>() * 4
}

/// Build a complete synthetic GGUF v3 binary for a tiny Qwen3 model.
///
/// Contains qwen2.* metadata keys and all 14 tensor types (3 global + 11 per layer)
/// with F32 dtype and deterministic data (0.01 * index).
fn build_qwen3_gguf(cfg: &TinyQwen3, include_output_weight: bool) -> Vec<u8> {
    let h = cfg.hidden as u64;
    let i = cfg.intermediate as u64;
    let v = cfg.vocab as u64;
    let nh = cfg.heads as u64;
    let nkv = cfg.kv_heads as u64;
    let hd = cfg.head_dim() as u64;

    // Define all tensors: (name, shape)
    let mut tensor_defs: Vec<(&str, Vec<u64>)> = vec![
        ("token_embd.weight", vec![v, h]),
        ("output_norm.weight", vec![h]),
    ];
    if include_output_weight {
        tensor_defs.push(("output.weight", vec![v, h]));
    }

    // Per-layer tensors for each layer
    for layer_idx in 0..cfg.layers {
        let prefix = format!("blk.{layer_idx}");
        let layer_tensors: Vec<(String, Vec<u64>)> = vec![
            (format!("{prefix}.attn_norm.weight"), vec![h]),
            (format!("{prefix}.attn_q.weight"), vec![nh * hd, h]),
            (format!("{prefix}.attn_k.weight"), vec![nkv * hd, h]),
            (format!("{prefix}.attn_v.weight"), vec![nkv * hd, h]),
            (format!("{prefix}.attn_output.weight"), vec![h, nh * hd]),
            (format!("{prefix}.attn_q_norm.weight"), vec![hd]),
            (format!("{prefix}.attn_k_norm.weight"), vec![hd]),
            (format!("{prefix}.ffn_norm.weight"), vec![h]),
            (format!("{prefix}.ffn_gate.weight"), vec![i, h]),
            (format!("{prefix}.ffn_up.weight"), vec![i, h]),
            (format!("{prefix}.ffn_down.weight"), vec![h, i]),
        ];
        for (name, shape) in layer_tensors {
            tensor_defs.push((Box::leak(name.into_boxed_str()), shape));
        }
    }

    let tensor_count = tensor_defs.len() as u64;

    // Metadata entries: general.architecture + 5 qwen2.* keys + rms_norm_eps + rope_theta = 8
    let metadata_count: u64 = 8;

    let mut buf = Vec::new();

    // --- Header ---
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes()); // version
    buf.extend_from_slice(&tensor_count.to_le_bytes());
    buf.extend_from_slice(&metadata_count.to_le_bytes());

    // --- Metadata ---
    meta_string(&mut buf, "general.architecture", "qwen2");
    meta_u32(&mut buf, "qwen2.embedding_length", cfg.hidden as u32);
    meta_u32(&mut buf, "qwen2.block_count", cfg.layers as u32);
    meta_u32(&mut buf, "qwen2.attention.head_count", cfg.heads as u32);
    meta_u32(
        &mut buf,
        "qwen2.attention.head_count_kv",
        cfg.kv_heads as u32,
    );
    meta_u32(
        &mut buf,
        "qwen2.feed_forward_length",
        cfg.intermediate as u32,
    );
    meta_f32(&mut buf, "qwen2.attention.layer_norm_rms_epsilon", 1e-6);
    meta_f32(&mut buf, "qwen2.rope.freq_base", 1_000_000.0);

    // --- Tensor info entries ---
    let mut offset: u64 = 0;
    for (name, shape) in &tensor_defs {
        tensor_info(&mut buf, name, shape, offset);
        offset += f32_byte_size(shape);
    }

    // --- Pad to 32-byte alignment ---
    while buf.len() % 32 != 0 {
        buf.push(0);
    }

    // --- Tensor data ---
    let mut elem_idx: u32 = 0;
    for (_name, shape) in &tensor_defs {
        let num_elements = shape.iter().product::<u64>() as u32;
        for _ in 0..num_elements {
            let val = 0.01 * (elem_idx % 100) as f32;
            buf.extend_from_slice(&val.to_le_bytes());
            elem_idx += 1;
        }
    }

    buf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_name_mapping_covers_all_qwen3_tensors() {
    // Verify all 14 tensor types map correctly (3 global + 11 per layer).
    let global = [
        ("token_embd.weight", "model.embed_tokens.weight"),
        ("output_norm.weight", "model.norm.weight"),
        ("output.weight", "lm_head.weight"),
    ];
    for (gguf, hf) in &global {
        assert_eq!(
            gguf_to_hf_name(gguf).as_deref(),
            Some(*hf),
            "global mapping failed for {gguf}"
        );
    }

    let layer_maps = [
        ("attn_norm.weight", "input_layernorm.weight"),
        ("attn_q.weight", "self_attn.q_proj.weight"),
        ("attn_k.weight", "self_attn.k_proj.weight"),
        ("attn_v.weight", "self_attn.v_proj.weight"),
        ("attn_output.weight", "self_attn.o_proj.weight"),
        ("attn_q_norm.weight", "self_attn.q_norm.weight"),
        ("attn_k_norm.weight", "self_attn.k_norm.weight"),
        ("ffn_norm.weight", "post_attention_layernorm.weight"),
        ("ffn_gate.weight", "mlp.gate_proj.weight"),
        ("ffn_up.weight", "mlp.up_proj.weight"),
        ("ffn_down.weight", "mlp.down_proj.weight"),
    ];
    for (gguf_suffix, hf_suffix) in &layer_maps {
        let gguf_name = format!("blk.0.{gguf_suffix}");
        let expected = format!("model.layers.0.{hf_suffix}");
        assert_eq!(
            gguf_to_hf_name(&gguf_name).as_deref(),
            Some(expected.as_str()),
            "layer mapping failed for {gguf_name}"
        );
    }
}

#[test]
fn test_parse_synthetic_qwen3_gguf() {
    let cfg = TinyQwen3::new();
    let data = build_qwen3_gguf(&cfg, true);
    let mut cursor = Cursor::new(&data);
    let file = GgufFile::read_from(&mut cursor).expect("should parse synthetic GGUF");

    assert_eq!(file.header.version, 3);
    assert_eq!(file.architecture(), Some("qwen2"));

    // 3 global + 11 per-layer tensors
    let expected_tensors = 3 + 11 * cfg.layers;
    assert_eq!(file.tensors.len(), expected_tensors);

    // Spot-check tensor shapes
    let embed = file.tensors.get("token_embd.weight").unwrap();
    assert_eq!(embed.shape, vec![cfg.vocab as u64, cfg.hidden as u64]);

    let q_proj = file.tensors.get("blk.0.attn_q.weight").unwrap();
    assert_eq!(
        q_proj.shape,
        vec![(cfg.heads * cfg.head_dim()) as u64, cfg.hidden as u64]
    );
}

#[test]
fn test_qwen3_config_from_gguf() {
    let cfg = TinyQwen3::new();
    let data = build_qwen3_gguf(&cfg, true);
    let mut cursor = Cursor::new(&data);
    let file = GgufFile::read_from(&mut cursor).unwrap();

    let qcfg = Qwen3GgufConfig::from_gguf(&file).expect("should extract config");
    assert_eq!(qcfg.hidden_size, cfg.hidden);
    assert_eq!(qcfg.intermediate_size, cfg.intermediate);
    assert_eq!(qcfg.num_hidden_layers, cfg.layers);
    assert_eq!(qcfg.num_attention_heads, cfg.heads);
    assert_eq!(qcfg.num_key_value_heads, cfg.kv_heads);
    assert_eq!(qcfg.head_dim, cfg.head_dim());
    assert!((qcfg.rms_norm_eps - 1e-6).abs() < 1e-12);
    assert!((qcfg.rope_theta - 1_000_000.0).abs() < 1.0);
}

#[test]
fn test_load_qwen3_tensors_from_synthetic_gguf() {
    let cfg = TinyQwen3::new();
    let data = build_qwen3_gguf(&cfg, true);
    let mut cursor = Cursor::new(&data);
    let file = GgufFile::read_from(&mut cursor).unwrap();

    let tensors = load_qwen3_tensors(&file, &mut cursor, false).expect("should load tensors");

    // All 14 tensors should be loaded with HF names
    let expected_hf_names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ];

    for name in &expected_hf_names {
        assert!(tensors.contains_key(*name), "missing tensor: {name}");
    }
    assert_eq!(tensors.len(), expected_hf_names.len());

    // Spot-check embed shape
    let embed = &tensors["model.embed_tokens.weight"];
    assert_eq!(embed.dims(), &[cfg.vocab, cfg.hidden]);

    // Verify all values are finite
    for (name, tensor) in &tensors {
        let data = tensor.to_vec1::<f32>().unwrap_or_else(|_| {
            // For 2D tensors, flatten
            let flat_len: usize = tensor.dims().iter().product();
            tensor
                .reshape([flat_len])
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        });
        for val in &data {
            assert!(val.is_finite(), "non-finite value in tensor {name}");
        }
    }
}

#[test]
fn test_tied_embeddings_creates_lm_head() {
    let cfg = TinyQwen3::new();
    // Build GGUF WITHOUT output.weight — lm_head should be cloned from embed
    let data = build_qwen3_gguf(&cfg, false);
    let mut cursor = Cursor::new(&data);
    let file = GgufFile::read_from(&mut cursor).unwrap();

    // Without tie_word_embeddings: no lm_head
    let tensors_untied = load_qwen3_tensors(&file, &mut Cursor::new(&data), false).unwrap();
    assert!(
        !tensors_untied.contains_key("lm_head.weight"),
        "lm_head should be absent without output.weight and tie=false"
    );

    // With tie_word_embeddings: lm_head cloned from embed
    let tensors_tied = load_qwen3_tensors(&file, &mut Cursor::new(&data), true).unwrap();
    assert!(
        tensors_tied.contains_key("lm_head.weight"),
        "lm_head should be present with tie=true"
    );
    assert_eq!(
        tensors_tied["lm_head.weight"].dims(),
        tensors_tied["model.embed_tokens.weight"].dims(),
        "tied lm_head should have same shape as embed_tokens"
    );
}

#[test]
fn test_gguf_to_varbuilder_to_qwen3_load() {
    // Full end-to-end: synthetic GGUF → load_qwen3_tensors → VarBuilder → Qwen3Model → forward
    let cfg = TinyQwen3::new();
    let data = build_qwen3_gguf(&cfg, true);
    let mut cursor = Cursor::new(&data);
    let file = GgufFile::read_from(&mut cursor).unwrap();

    let tensors = load_qwen3_tensors(&file, &mut cursor, false).expect("should load tensors");

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    let qwen3_cfg = Qwen3Config::new(
        cfg.hidden,       // hidden_size
        cfg.intermediate, // intermediate_size
        cfg.layers,       // num_hidden_layers
        cfg.heads,        // num_attention_heads
        cfg.kv_heads,     // num_key_value_heads
        cfg.vocab,        // vocab_size
        1e-6,             // rms_norm_eps
        1_000_000.0,      // rope_theta
        32768,            // max_position_embeddings
        false,            // tie_word_embeddings
        None,             // rope_scaling
    );

    let model = nn_qwen3::Qwen3Model::load(&vb, qwen3_cfg)
        .expect("should load Qwen3Model from GGUF-sourced VarBuilder");

    // Forward pass: single token, single position
    // Qwen3 returns [batch, seq_len, vocab_size] = [1, 1, vocab_size]
    let logits = model.forward(&[0], &[0]).expect("forward should succeed");
    assert_eq!(
        logits.dims(),
        &[1, 1, cfg.vocab],
        "logits shape should be [1, 1, vocab_size]"
    );

    // All logits should be finite (no NaN/Inf from the pipeline)
    let logits_data = logits.to_vec3::<f32>().expect("should extract logits");
    for val in &logits_data[0][0] {
        assert!(val.is_finite(), "logit is not finite: {val}");
    }
}

#[test]
fn test_real_qwen3_gguf_loading() {
    // Gated behind QWEN3_GGUF_PATH env var. Skips gracefully when unset.
    let path = match std::env::var("QWEN3_GGUF_PATH") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("QWEN3_GGUF_PATH not set, skipping real GGUF test");
            return;
        }
    };

    let file = GgufFile::open(&path).expect("should open real GGUF file");

    // Should be qwen2 or qwen3 architecture
    let arch = file.architecture().expect("should have architecture");
    assert!(
        arch == "qwen2" || arch == "qwen3",
        "unexpected architecture: {arch}"
    );

    // Extract config
    let qcfg =
        Qwen3GgufConfig::from_gguf(&file).expect("should extract Qwen3 config from real GGUF");
    assert!(qcfg.hidden_size > 0);
    assert!(qcfg.num_hidden_layers > 0);
    assert_eq!(qcfg.head_dim, qcfg.hidden_size / qcfg.num_attention_heads);

    eprintln!(
        "Real Qwen3 GGUF: hidden={}, layers={}, heads={}, kv_heads={}, vocab={}",
        qcfg.hidden_size,
        qcfg.num_hidden_layers,
        qcfg.num_attention_heads,
        qcfg.num_key_value_heads,
        qcfg.vocab_size,
    );
}
