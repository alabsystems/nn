// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: full GGUF execution pipeline with synthetic weights.
//!
//! Constructs a synthetic GGUF binary in memory for a tiny 2-layer Llama
//! model, parses it with `GgufFile::read_from()`, builds a weighted
//! computation graph via `to_computation_graph_with_weights()`, and
//! compiles through `compile_trace_to_plan_with_fusion()`.
//!
//! This tests the end-to-end pipeline without requiring a real GGUF file.

use std::io::Cursor;

use nn_gguf::{GgufDType, GgufFile};

// ---------------------------------------------------------------------------
// Tiny model parameters
// ---------------------------------------------------------------------------

const VOCAB: usize = 32;
const HIDDEN: usize = 64;
const NUM_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16; // HIDDEN / NUM_HEADS
const FFN_DIM: usize = 128;
const NUM_LAYERS: usize = 2;
const RMS_NORM_EPS: f32 = 1e-5;
const ROPE_BASE: f32 = 10000.0;
const MAX_SEQ_LEN: u32 = 128;

// ---------------------------------------------------------------------------
// GGUF binary builder helpers
// ---------------------------------------------------------------------------

/// Writer that accumulates a GGUF v3 binary in memory.
struct GgufBuilder {
    /// Accumulated header + metadata + tensor info bytes.
    header_buf: Vec<u8>,
    /// Tensor data blobs with their declared offsets.
    tensor_data: Vec<u8>,
    /// Running byte offset into the data section.
    data_cursor: u64,
    /// Number of tensors registered.
    tensor_count: u64,
    /// Metadata entries (buffered so we can write count into header).
    metadata_entries: Vec<Vec<u8>>,
}

impl GgufBuilder {
    fn new() -> Self {
        Self {
            header_buf: Vec::new(),
            tensor_data: Vec::new(),
            data_cursor: 0,
            tensor_count: 0,
            metadata_entries: Vec::new(),
        }
    }

    // -- metadata helpers --

    fn add_meta_string(&mut self, key: &str, value: &str) {
        let mut buf = Vec::new();
        write_string(&mut buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // type_id = STRING
        write_string(&mut buf, value);
        self.metadata_entries.push(buf);
    }

    fn add_meta_u32(&mut self, key: &str, value: u32) {
        let mut buf = Vec::new();
        write_string(&mut buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // type_id = UINT32
        buf.extend_from_slice(&value.to_le_bytes());
        self.metadata_entries.push(buf);
    }

    fn add_meta_f32(&mut self, key: &str, value: f32) {
        let mut buf = Vec::new();
        write_string(&mut buf, key);
        buf.extend_from_slice(&6u32.to_le_bytes()); // type_id = FLOAT32
        buf.extend_from_slice(&value.to_le_bytes());
        self.metadata_entries.push(buf);
    }

    // -- tensor registration --

    /// Register a tensor with F32 data. `shape` is in GGUF order.
    fn add_f32_tensor(&mut self, name: &str, shape: &[u64], data: &[f32]) {
        let byte_len = data.len() * 4;
        // Align data_cursor to 32 bytes for each tensor.
        let aligned = align_up(self.data_cursor, 32);
        let padding = (aligned - self.data_cursor) as usize;
        self.tensor_data
            .extend(std::iter::repeat_n(0u8, padding));
        self.data_cursor = aligned;

        let mut info_buf = Vec::new();
        write_string(&mut info_buf, name);
        info_buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &dim in shape {
            info_buf.extend_from_slice(&dim.to_le_bytes());
        }
        info_buf.extend_from_slice(&(GgufDType::F32 as u32).to_le_bytes());
        info_buf.extend_from_slice(&self.data_cursor.to_le_bytes());
        self.header_buf.extend_from_slice(&info_buf);

        // Append raw f32 data.
        for &v in data {
            self.tensor_data.extend_from_slice(&v.to_le_bytes());
        }
        self.data_cursor += byte_len as u64;
        self.tensor_count += 1;
    }

    /// Finalize into a complete GGUF binary.
    fn build(self) -> Vec<u8> {
        let mut out = Vec::new();
        // -- header --
        let magic: u32 = 0x4647_5547; // "GGUF"
        out.extend_from_slice(&magic.to_le_bytes());
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&self.tensor_count.to_le_bytes());
        out.extend_from_slice(&(self.metadata_entries.len() as u64).to_le_bytes());

        // -- metadata --
        for entry in &self.metadata_entries {
            out.extend_from_slice(entry);
        }

        // -- tensor info (already serialized in header_buf) --
        out.extend_from_slice(&self.header_buf);

        // -- pad to 32-byte alignment --
        let alignment = 32u64;
        let data_offset = align_up(out.len() as u64, alignment) as usize;
        out.resize(data_offset, 0);

        // -- tensor data --
        out.extend_from_slice(&self.tensor_data);

        out
    }
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn align_up(val: u64, alignment: u64) -> u64 {
    val.div_ceil(alignment) * alignment
}

// ---------------------------------------------------------------------------
// Synthetic weight generation
// ---------------------------------------------------------------------------

/// Generate deterministic synthetic weight data for a given shape.
fn synthetic_weights(shape: &[usize]) -> Vec<f32> {
    let n: usize = shape.iter().product();
    // Small values to avoid numerical issues during compilation.
    (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect()
}

/// Build a complete synthetic GGUF binary for a tiny 2-layer Llama model.
fn build_synthetic_gguf() -> Vec<u8> {
    let mut b = GgufBuilder::new();

    // -- metadata: Llama architecture keys --
    b.add_meta_string("general.architecture", "llama");
    b.add_meta_string("general.name", "tiny-llama-test");
    b.add_meta_u32("llama.embedding_length", HIDDEN as u32);
    b.add_meta_u32("llama.block_count", NUM_LAYERS as u32);
    b.add_meta_u32("llama.attention.head_count", NUM_HEADS as u32);
    b.add_meta_u32("llama.attention.head_count_kv", NUM_KV_HEADS as u32);
    b.add_meta_u32("llama.feed_forward_length", FFN_DIM as u32);
    b.add_meta_u32("llama.context_length", MAX_SEQ_LEN);
    b.add_meta_u32("llama.vocab_size", VOCAB as u32);
    b.add_meta_f32("llama.attention.layer_norm_rms_epsilon", RMS_NORM_EPS);
    b.add_meta_f32("llama.rope.freq_base", ROPE_BASE);

    // -- global weights --
    let embd_shape = [VOCAB, HIDDEN];
    b.add_f32_tensor(
        "token_embd.weight",
        &embd_shape.map(|d| d as u64),
        &synthetic_weights(&embd_shape),
    );

    // -- per-layer weights --
    for layer in 0..NUM_LAYERS {
        // Attention norm
        let norm_shape = [HIDDEN];
        b.add_f32_tensor(
            &format!("blk.{layer}.attn_norm.weight"),
            &norm_shape.map(|d| d as u64),
            &synthetic_weights(&norm_shape),
        );

        // Q, K, V projections
        let q_shape = [NUM_HEADS * HEAD_DIM, HIDDEN];
        b.add_f32_tensor(
            &format!("blk.{layer}.attn_q.weight"),
            &q_shape.map(|d| d as u64),
            &synthetic_weights(&q_shape),
        );

        let kv_shape = [NUM_KV_HEADS * HEAD_DIM, HIDDEN];
        b.add_f32_tensor(
            &format!("blk.{layer}.attn_k.weight"),
            &kv_shape.map(|d| d as u64),
            &synthetic_weights(&kv_shape),
        );
        b.add_f32_tensor(
            &format!("blk.{layer}.attn_v.weight"),
            &kv_shape.map(|d| d as u64),
            &synthetic_weights(&kv_shape),
        );

        // Output projection
        let o_shape = [HIDDEN, NUM_HEADS * HEAD_DIM];
        b.add_f32_tensor(
            &format!("blk.{layer}.attn_output.weight"),
            &o_shape.map(|d| d as u64),
            &synthetic_weights(&o_shape),
        );

        // FFN norm
        b.add_f32_tensor(
            &format!("blk.{layer}.ffn_norm.weight"),
            &norm_shape.map(|d| d as u64),
            &synthetic_weights(&norm_shape),
        );

        // FFN gate, up, down projections
        let gate_shape = [FFN_DIM, HIDDEN];
        b.add_f32_tensor(
            &format!("blk.{layer}.ffn_gate.weight"),
            &gate_shape.map(|d| d as u64),
            &synthetic_weights(&gate_shape),
        );
        b.add_f32_tensor(
            &format!("blk.{layer}.ffn_up.weight"),
            &gate_shape.map(|d| d as u64),
            &synthetic_weights(&gate_shape),
        );

        let down_shape = [HIDDEN, FFN_DIM];
        b.add_f32_tensor(
            &format!("blk.{layer}.ffn_down.weight"),
            &down_shape.map(|d| d as u64),
            &synthetic_weights(&down_shape),
        );
    }

    // Final output norm
    let norm_shape = [HIDDEN];
    b.add_f32_tensor(
        "output_norm.weight",
        &norm_shape.map(|d| d as u64),
        &synthetic_weights(&norm_shape),
    );

    // LM head (output.weight) — same shape as embedding
    let lm_shape = [VOCAB, HIDDEN];
    b.add_f32_tensor(
        "output.weight",
        &lm_shape.map(|d| d as u64),
        &synthetic_weights(&lm_shape),
    );

    b.build()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_synthetic_gguf_parses() {
    let data = build_synthetic_gguf();
    let mut cursor = Cursor::new(&data);
    let gguf = GgufFile::read_from(&mut cursor).expect("should parse synthetic GGUF");

    assert_eq!(gguf.header.version, 3);
    assert_eq!(gguf.architecture(), Some("llama"));
    assert_eq!(gguf.model_name(), Some("tiny-llama-test"));

    // Per layer: attn_norm, attn_q, attn_k, attn_v, attn_output,
    // ffn_norm, ffn_gate, ffn_up, ffn_down = 9 weight tensors.
    // Global: token_embd, output_norm, output = 3 tensors.
    let expected_tensors = NUM_LAYERS * 9 + 3;
    assert_eq!(
        gguf.tensors.len(),
        expected_tensors,
        "expected {expected_tensors} tensors, got {}",
        gguf.tensors.len()
    );
}

#[test]
fn test_synthetic_gguf_config_extraction() {
    let data = build_synthetic_gguf();
    let mut cursor = Cursor::new(&data);
    let gguf = GgufFile::read_from(&mut cursor).expect("parse");

    let config = nn_gguf::LlamaConfig::from_gguf(&gguf).expect("config extraction");
    assert_eq!(config.vocab_size, VOCAB);
    assert_eq!(config.hidden_dim, HIDDEN);
    assert_eq!(config.num_layers, NUM_LAYERS);
    assert_eq!(config.num_heads, NUM_HEADS);
    assert_eq!(config.num_kv_heads, NUM_KV_HEADS);
    assert_eq!(config.head_dim, HEAD_DIM);
    assert_eq!(config.ffn_dim, FFN_DIM);
    assert!((config.rms_norm_eps - RMS_NORM_EPS).abs() < 1e-10);
    assert!((config.rope_base - ROPE_BASE).abs() < 1.0);
    assert_eq!(config.max_seq_len, MAX_SEQ_LEN as usize);
}

#[test]
fn test_synthetic_gguf_weight_dequant() {
    let data = build_synthetic_gguf();
    let mut cursor = Cursor::new(&data);
    let gguf = GgufFile::read_from(&mut cursor).expect("parse");

    // Read back the embedding weight and verify roundtrip.
    let (embd_data, embd_shape) = gguf
        .read_tensor_f32(&mut cursor, "token_embd.weight")
        .expect("should read embedding");

    assert_eq!(embd_shape, vec![VOCAB, HIDDEN]);
    assert_eq!(embd_data.len(), VOCAB * HIDDEN);

    // Verify against our synthetic generation.
    let expected = synthetic_weights(&[VOCAB, HIDDEN]);
    for (i, (&got, &want)) in embd_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "embedding mismatch at index {i}: got {got}, want {want}"
        );
    }
}

#[test]
fn test_synthetic_gguf_builds_weighted_graph() {
    let data = build_synthetic_gguf();
    let mut cursor = Cursor::new(&data);
    let gguf = GgufFile::read_from(&mut cursor).expect("parse");

    let graph = gguf
        .to_computation_graph_with_weights(&mut cursor)
        .expect("should build weighted graph");

    // Same node count as the shape-only graph.
    let expected_nodes = 4 + 22 * NUM_LAYERS;
    assert_eq!(
        graph.len(),
        expected_nodes,
        "expected {expected_nodes} nodes, got {}",
        graph.len()
    );

    // Output shape: [batch=1, seq=1, vocab]
    let output = graph.output_node().expect("should have output");
    assert_eq!(output.output_shape(), &[1, 1, VOCAB]);

    // Topology is valid.
    graph
        .validate_topology()
        .expect("weighted graph should have valid topology");
}

#[test]
fn test_synthetic_gguf_compiles_with_fusion() {
    let data = build_synthetic_gguf();
    let mut cursor = Cursor::new(&data);
    let gguf = GgufFile::read_from(&mut cursor).expect("parse");

    let graph = gguf
        .to_computation_graph_with_weights(&mut cursor)
        .expect("weighted graph");

    // Compile through the full fusion pipeline.
    let plan = nn_dsl::trace_compile::compile_trace_to_plan_with_fusion(&graph)
        .expect("should compile weighted graph through fusion pipeline");

    // Plan must have steps.
    assert!(
        !plan.steps.is_empty(),
        "compiled plan should have at least one step"
    );

    // Output step is valid.
    assert_eq!(
        plan.output_step,
        plan.steps.len() - 1,
        "output_step should be the last step"
    );

    // Input shape should be [1, 1] (token_ids: [batch, seq]).
    assert_eq!(plan.input_shapes.len(), 1, "should have exactly one input");
    assert_eq!(
        plan.input_shapes[0],
        vec![1, 1],
        "input shape should be [batch=1, seq=1]"
    );

    // Weight names should be populated (weighted graph has real data).
    assert!(
        !plan.weight_names.is_empty(),
        "compiled plan should reference weight names"
    );

    println!(
        "Compiled plan: {} steps, {} weight names, output_step={}",
        plan.steps.len(),
        plan.weight_names.len(),
        plan.output_step
    );
}

#[test]
fn test_synthetic_gguf_fusion_reduces_dispatches() {
    let data = build_synthetic_gguf();
    let mut cursor = Cursor::new(&data);
    let gguf = GgufFile::read_from(&mut cursor).expect("parse");

    let graph = gguf
        .to_computation_graph_with_weights(&mut cursor)
        .expect("weighted graph");

    // Compile without fusion.
    let plan_no_fusion =
        nn_dsl::trace_compile::compile_trace_to_plan(&graph).expect("compile without fusion");

    // Compile with fusion.
    let plan_fused = nn_dsl::trace_compile::compile_trace_to_plan_with_fusion(&graph)
        .expect("compile with fusion");

    // Fused plan should have same or fewer steps.
    assert!(
        plan_fused.steps.len() <= plan_no_fusion.steps.len(),
        "fusion should not increase step count: fused={}, unfused={}",
        plan_fused.steps.len(),
        plan_no_fusion.steps.len()
    );

    // Partition analysis should report reduction.
    let (pre, post) = nn_dsl::trace_compile::partition_analysis(&graph);
    assert!(pre > 0, "should have pre-partition dispatches");
    assert!(
        post <= pre,
        "fusion should reduce dispatches: pre={pre}, post={post}"
    );

    println!(
        "Dispatch reduction: {} unfused → {} fused ({:.1}% reduction)",
        plan_no_fusion.steps.len(),
        plan_fused.steps.len(),
        (1.0 - plan_fused.steps.len() as f64 / plan_no_fusion.steps.len() as f64) * 100.0
    );
    println!("Partition analysis: {pre} pre → {post} post");
}

#[test]
fn test_synthetic_gguf_tied_embeddings() {
    // Build a GGUF that omits output.weight to exercise the tied-embedding
    // fallback path (reuses token_embd.weight for the LM head).
    let mut b = GgufBuilder::new();

    b.add_meta_string("general.architecture", "llama");
    b.add_meta_u32("llama.embedding_length", HIDDEN as u32);
    b.add_meta_u32("llama.block_count", 1);
    b.add_meta_u32("llama.attention.head_count", NUM_HEADS as u32);
    b.add_meta_u32("llama.attention.head_count_kv", NUM_KV_HEADS as u32);
    b.add_meta_u32("llama.feed_forward_length", FFN_DIM as u32);
    b.add_meta_u32("llama.context_length", MAX_SEQ_LEN);
    b.add_meta_u32("llama.vocab_size", VOCAB as u32);

    // Global weights.
    let embd = synthetic_weights(&[VOCAB, HIDDEN]);
    b.add_f32_tensor("token_embd.weight", &[VOCAB as u64, HIDDEN as u64], &embd);

    // Single layer weights.
    let norm = synthetic_weights(&[HIDDEN]);
    b.add_f32_tensor("blk.0.attn_norm.weight", &[HIDDEN as u64], &norm);
    b.add_f32_tensor(
        "blk.0.attn_q.weight",
        &[(NUM_HEADS * HEAD_DIM) as u64, HIDDEN as u64],
        &synthetic_weights(&[NUM_HEADS * HEAD_DIM, HIDDEN]),
    );
    b.add_f32_tensor(
        "blk.0.attn_k.weight",
        &[(NUM_KV_HEADS * HEAD_DIM) as u64, HIDDEN as u64],
        &synthetic_weights(&[NUM_KV_HEADS * HEAD_DIM, HIDDEN]),
    );
    b.add_f32_tensor(
        "blk.0.attn_v.weight",
        &[(NUM_KV_HEADS * HEAD_DIM) as u64, HIDDEN as u64],
        &synthetic_weights(&[NUM_KV_HEADS * HEAD_DIM, HIDDEN]),
    );
    b.add_f32_tensor(
        "blk.0.attn_output.weight",
        &[HIDDEN as u64, (NUM_HEADS * HEAD_DIM) as u64],
        &synthetic_weights(&[HIDDEN, NUM_HEADS * HEAD_DIM]),
    );
    b.add_f32_tensor("blk.0.ffn_norm.weight", &[HIDDEN as u64], &norm);
    b.add_f32_tensor(
        "blk.0.ffn_gate.weight",
        &[FFN_DIM as u64, HIDDEN as u64],
        &synthetic_weights(&[FFN_DIM, HIDDEN]),
    );
    b.add_f32_tensor(
        "blk.0.ffn_up.weight",
        &[FFN_DIM as u64, HIDDEN as u64],
        &synthetic_weights(&[FFN_DIM, HIDDEN]),
    );
    b.add_f32_tensor(
        "blk.0.ffn_down.weight",
        &[HIDDEN as u64, FFN_DIM as u64],
        &synthetic_weights(&[HIDDEN, FFN_DIM]),
    );
    b.add_f32_tensor("output_norm.weight", &[HIDDEN as u64], &norm);

    // NOTE: no "output.weight" tensor — should fall back to token_embd.weight.

    let data = b.build();
    let mut cursor = Cursor::new(&data);
    let gguf = GgufFile::read_from(&mut cursor).expect("parse tied-embedding GGUF");

    let graph = gguf
        .to_computation_graph_with_weights(&mut cursor)
        .expect("should build graph with tied embeddings");

    // Should still compile.
    let plan = nn_dsl::trace_compile::compile_trace_to_plan_with_fusion(&graph)
        .expect("tied-embedding graph should compile");

    assert!(!plan.steps.is_empty());

    let output = graph.output_node().expect("output node");
    assert_eq!(output.output_shape(), &[1, 1, VOCAB]);

    println!(
        "Tied-embedding model compiled: {} steps, {} weight names",
        plan.steps.len(),
        plan.weight_names.len()
    );
}
