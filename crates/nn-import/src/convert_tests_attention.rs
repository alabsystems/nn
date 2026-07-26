// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Transformer multi-head attention convert tests: Linear Q/K/V + Reshape +
//! Transpose + ScaledDotProductAttention + Transpose + Reshape + Linear.
//!
//! Models a synthetic 2-head self-attention subgraph found in dpdf models
//! (Table Transformer, Granite-Docling, etc.):
//!   Input [1, 4, 16] -> Q/K/V projections [1, 4, 16]
//!   -> Reshape [1, 4, 2, 8] -> Transpose [1, 2, 4, 8]
//!   -> SDPA [1, 2, 4, 8] -> Transpose [1, 4, 2, 8]
//!   -> Reshape [1, 4, 16] -> Output projection [1, 4, 16]

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use nn_core::dyn_tensor::trace::TraceOp;

use crate::graph_build::ImportedGraph;
use crate::import_model;

static ATTN_FIXTURE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Write synthetic 2-head attention weights to a safetensors file.
///
/// Q/K/V projections: weight [16, 16] = 256 elements each, bias [16]
/// Output projection: weight [16, 16] = 256 elements, bias [16]
fn write_attention_weights(dir: &Path) -> std::path::PathBuf {
    let mut tensors = HashMap::new();

    // Weight matrices: [16, 16] = 256 elements each.
    let q_w: Vec<u8> = (0..256)
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();
    let k_w: Vec<u8> = (0..256)
        .flat_map(|i| (((i + 256) as f32) * 0.001).to_le_bytes())
        .collect();
    let v_w: Vec<u8> = (0..256)
        .flat_map(|i| (((i + 512) as f32) * 0.001).to_le_bytes())
        .collect();
    let out_w: Vec<u8> = (0..256)
        .flat_map(|i| (((i + 768) as f32) * 0.001).to_le_bytes())
        .collect();

    // Bias vectors: [16] = 16 elements each (all zeros).
    let zero_bias: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();

    for (name, shape, data) in [
        ("attn.q_proj.weight", vec![16, 16], q_w.as_slice()),
        ("attn.q_proj.bias", vec![16], zero_bias.as_slice()),
        ("attn.k_proj.weight", vec![16, 16], k_w.as_slice()),
        ("attn.k_proj.bias", vec![16], zero_bias.as_slice()),
        ("attn.v_proj.weight", vec![16, 16], v_w.as_slice()),
        ("attn.v_proj.bias", vec![16], zero_bias.as_slice()),
        ("attn.out_proj.weight", vec![16, 16], out_w.as_slice()),
        ("attn.out_proj.bias", vec![16], zero_bias.as_slice()),
    ] {
        tensors.insert(
            name.to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, data).unwrap(),
        );
    }

    let weights_path = dir.join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

/// Import the 2-head attention mini fixture from disk.
fn import_attention_fixture() -> ImportedGraph {
    let id = ATTN_FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("nn_import_attn_{}_{}", std::process::id(), id));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/attention_2head_mini.json"),
    )
    .unwrap();
    let weights_path = write_attention_weights(&dir);
    let imported = import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    imported
}

// ---------------------------------------------------------------------------
// Graph structure tests (no Metal required)
// ---------------------------------------------------------------------------

/// E2E: 2-head attention subgraph imports with correct structure.
///
/// Exercises the full import pipeline: parse JSON -> weight load -> FQN mapping ->
/// op mapping -> graph build -> topology validation.
///
/// This models the attention pattern from dpdf Transformer models:
/// Linear(Q) + Linear(K) + Linear(V) -> Reshape -> Transpose -> SDPA ->
/// Transpose -> Reshape -> Linear(out_proj)
#[test]
fn test_import_attention_2head_structure() {
    let imported = import_attention_fixture();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["out_proj"]);

    // 1 Input + 8 params + 13 compute ops = 22 total nodes.
    assert_eq!(imported.graph.len(), 22);

    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::Linear { .. }),
        "expected Linear as output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 4, 16]);
}

/// E2E: all attention-specific aten ops map to correct TraceOp variants.
#[test]
fn test_import_attention_2head_op_counts() {
    let imported = import_attention_fixture();
    let nodes = imported.graph.nodes();
    let count = |pred: fn(&TraceOp) -> bool| nodes.iter().filter(|n| pred(n.op())).count();

    // 4 Linear: Q, K, V projections + output projection
    assert_eq!(count(|op| matches!(op, TraceOp::Linear { .. })), 4);
    // 4 Reshape: Q/K/V split to heads + merge heads back
    assert_eq!(count(|op| matches!(op, TraceOp::Reshape { .. })), 4);
    // 4 Transpose: Q/K/V to [B, H, S, D] + attention output back to [B, S, H, D]
    assert_eq!(count(|op| matches!(op, TraceOp::Transpose { .. })), 4);
    // 1 SDPA
    assert_eq!(count(|op| matches!(op, TraceOp::Sdpa { .. })), 1);
}

/// E2E: intermediate shapes propagate correctly through multi-head attention.
#[test]
fn test_import_attention_2head_shapes() {
    let imported = import_attention_fixture();
    let nodes = imported.graph.nodes();

    // Input: [1, 4, 16]
    let input = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Input))
        .unwrap();
    assert_eq!(input.output_shape(), &[1, 4, 16]);

    // Q/K/V linear outputs: [1, 4, 16]
    let q_linear = nodes.iter().find(|n| n.name() == "q_linear").unwrap();
    assert_eq!(q_linear.output_shape(), &[1, 4, 16]);

    // Q/K/V reshaped to multi-head: [1, 4, 2, 8]
    let q_reshape = nodes.iter().find(|n| n.name() == "q_reshape").unwrap();
    assert_eq!(q_reshape.output_shape(), &[1, 4, 2, 8]);

    // Q/K/V transposed for attention: [1, 2, 4, 8] (batch, heads, seq, head_dim)
    let q_transposed = nodes.iter().find(|n| n.name() == "q_transposed").unwrap();
    assert_eq!(q_transposed.output_shape(), &[1, 2, 4, 8]);

    // SDPA output: [1, 2, 4, 8]
    let attn_out = nodes.iter().find(|n| n.name() == "attn_out").unwrap();
    assert_eq!(attn_out.output_shape(), &[1, 2, 4, 8]);

    // Transposed back: [1, 4, 2, 8]
    let attn_transposed = nodes
        .iter()
        .find(|n| n.name() == "attn_transposed")
        .unwrap();
    assert_eq!(attn_transposed.output_shape(), &[1, 4, 2, 8]);

    // Flattened: [1, 4, 16]
    let attn_flat = nodes.iter().find(|n| n.name() == "attn_flat").unwrap();
    assert_eq!(attn_flat.output_shape(), &[1, 4, 16]);

    // Output projection: [1, 4, 16]
    let out_proj = nodes.iter().find(|n| n.name() == "out_proj").unwrap();
    assert_eq!(out_proj.output_shape(), &[1, 4, 16]);
}

/// E2E: SDPA scale parameter survives import.
///
/// scale = 1/sqrt(head_dim) = 1/sqrt(8) ~= 0.3536
#[test]
fn test_import_attention_2head_sdpa_params() {
    let imported = import_attention_fixture();
    let nodes = imported.graph.nodes();

    let sdpa_node = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Sdpa { .. }))
        .expect("must have Sdpa node");

    if let TraceOp::Sdpa { scale } = sdpa_node.op() {
        let expected = 1.0 / (8.0_f64).sqrt();
        assert!(
            (*scale - expected).abs() < 1e-6,
            "expected scale={expected:.6}, got {scale:.6}"
        );
    }
}

/// E2E: SDPA node has correct input dependencies (Q, K, V tensors).
#[test]
fn test_import_attention_2head_sdpa_inputs() {
    let imported = import_attention_fixture();
    let nodes = imported.graph.nodes();

    let sdpa_node = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Sdpa { .. }))
        .expect("must have Sdpa node");

    // SDPA should have exactly 3 inputs (Q, K, V).
    assert_eq!(
        sdpa_node.inputs().len(),
        3,
        "SDPA expects 3 inputs (Q, K, V)"
    );

    // All inputs should be Transpose nodes (the Q/K/V transposed tensors).
    for &input_id in sdpa_node.inputs() {
        let input_node = nodes.iter().find(|n| n.id() == input_id).unwrap();
        assert!(
            matches!(input_node.op(), TraceOp::Transpose { dim0: 1, dim1: 2 }),
            "SDPA input should be Transpose(1,2), got: {:?}",
            input_node.op()
        );
    }
}

/// E2E: Transpose parameters are correct for multi-head split/merge.
#[test]
fn test_import_attention_2head_transpose_params() {
    let imported = import_attention_fixture();
    let nodes = imported.graph.nodes();

    let transpose_nodes: Vec<_> = nodes
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Transpose { .. }))
        .collect();
    assert_eq!(transpose_nodes.len(), 4);

    // All should transpose dims 1 and 2 (seq <-> heads).
    for tn in &transpose_nodes {
        if let TraceOp::Transpose { dim0, dim1 } = tn.op() {
            assert_eq!(*dim0, 1, "dim0 should be 1");
            assert_eq!(*dim1, 2, "dim1 should be 2");
        }
    }
}
