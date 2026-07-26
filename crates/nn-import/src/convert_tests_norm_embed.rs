// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Convert tests for Embedding, LayerNorm, Softmax, and GroupNorm ops.
//!
//! Exercises the op_map paths for:
//! - `aten::embedding` — lookup table: weight[vocab, dim] x indices[batch, seq] -> [batch, seq, dim]
//! - `aten::layer_norm` — LayerNorm with weight/bias
//! - `aten::softmax` — softmax along specified dimension
//! - `aten::group_norm` — GroupNorm with weight/bias, num_groups

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use nn_core::dyn_tensor::trace::TraceOp;

use crate::graph_build::ImportedGraph;
use crate::import_model;

static NORM_EMBED_COUNTER: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// Weight helpers
// ---------------------------------------------------------------------------

/// Write synthetic embedding weights: embed.weight [10, 8] = 80 elements.
fn write_embedding_weights(dir: &Path) -> std::path::PathBuf {
    let embed_w: Vec<u8> = (0..80)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "embed.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![10, 8], &embed_w)
            .unwrap(),
    );
    let weights_path = dir.join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

/// Write synthetic LayerNorm weights: ln.weight [16], ln.bias [16].
fn write_layernorm_softmax_weights(dir: &Path) -> std::path::PathBuf {
    let ln_w: Vec<u8> = [1.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let ln_b: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "ln.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16], &ln_w).unwrap(),
    );
    tensors.insert(
        "ln.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16], &ln_b).unwrap(),
    );
    let weights_path = dir.join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

/// Write synthetic GroupNorm weights: gn.weight [16], gn.bias [16].
fn write_groupnorm_weights(dir: &Path) -> std::path::PathBuf {
    let gn_w: Vec<u8> = [1.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let gn_b: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut tensors = HashMap::new();
    tensors.insert(
        "gn.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16], &gn_w).unwrap(),
    );
    tensors.insert(
        "gn.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![16], &gn_b).unwrap(),
    );
    let weights_path = dir.join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

// ---------------------------------------------------------------------------
// Fixture importers
// ---------------------------------------------------------------------------

fn make_temp_dir(label: &str) -> std::path::PathBuf {
    let id = NORM_EMBED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("nn_import_{label}_{}_{}", std::process::id(), id));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn import_embedding_fixture() -> ImportedGraph {
    let dir = make_temp_dir("embed");
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/embedding_lookup.json"),
    )
    .unwrap();
    let weights_path = write_embedding_weights(&dir);
    let imported = import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    imported
}

fn import_layernorm_softmax_fixture() -> ImportedGraph {
    let dir = make_temp_dir("lnsm");
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/layernorm_softmax.json"),
    )
    .unwrap();
    let weights_path = write_layernorm_softmax_weights(&dir);
    let imported = import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    imported
}

fn import_groupnorm_fixture() -> ImportedGraph {
    let dir = make_temp_dir("gn");
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/groupnorm_block.json"),
    )
    .unwrap();
    let weights_path = write_groupnorm_weights(&dir);
    let imported = import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    imported
}

// ---------------------------------------------------------------------------
// Embedding tests
// ---------------------------------------------------------------------------

/// E2E: Embedding lookup graph imports with correct structure.
///
/// Graph: embedding.default(weight[10,8], indices[1,4]) -> [1,4,8]
#[test]
fn test_import_embedding_lookup_structure() {
    let imported = import_embedding_fixture();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["indices"]);
    assert_eq!(imported.output_names, vec!["embedding"]);

    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::Embedding { .. }),
        "expected Embedding as output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 4, 8]);
}

/// E2E: Embedding weight shape survives import.
#[test]
fn test_import_embedding_lookup_weight_shape() {
    let imported = import_embedding_fixture();
    let nodes = imported.graph.nodes();

    let embed = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Embedding { .. }))
        .unwrap();
    if let TraceOp::Embedding { weight } = embed.op() {
        assert_eq!(
            weight.shape(),
            &[10, 8],
            "embedding weight should be [vocab=10, dim=8]"
        );
    }
}

/// E2E: Embedding op count is exactly 1.
#[test]
fn test_import_embedding_lookup_op_counts() {
    let imported = import_embedding_fixture();
    let nodes = imported.graph.nodes();
    let count = nodes
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Embedding { .. }))
        .count();
    assert_eq!(count, 1);
}

// ---------------------------------------------------------------------------
// LayerNorm -> Softmax tests
// ---------------------------------------------------------------------------

/// E2E: LayerNorm -> Softmax graph imports with correct structure.
///
/// Graph: layer_norm(x[1,4,16], w[16], b[16]) -> softmax(dim=-1) -> [1,4,16]
#[test]
fn test_import_layernorm_softmax_structure() {
    let imported = import_layernorm_softmax_fixture();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["softmax"]);

    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::Softmax { .. }),
        "expected Softmax as output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 4, 16]);
}

/// E2E: LayerNorm + Softmax op counts.
#[test]
fn test_import_layernorm_softmax_op_counts() {
    let imported = import_layernorm_softmax_fixture();
    let nodes = imported.graph.nodes();
    let count = |pred: fn(&TraceOp) -> bool| nodes.iter().filter(|n| pred(n.op())).count();

    assert_eq!(count(|op| matches!(op, TraceOp::LayerNorm { .. })), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::Softmax { .. })), 1);
}

/// E2E: LayerNorm parameters (eps, weight shape) survive import.
#[test]
fn test_import_layernorm_softmax_params() {
    let imported = import_layernorm_softmax_fixture();
    let nodes = imported.graph.nodes();

    let ln = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::LayerNorm { .. }))
        .unwrap();
    if let TraceOp::LayerNorm { eps, weight, bias } = ln.op() {
        assert!((*eps - 1e-5).abs() < 1e-8, "expected eps=1e-5, got {eps}");
        assert_eq!(weight.shape(), &[16], "LayerNorm weight shape");
        assert_eq!(bias.shape(), &[16], "LayerNorm bias shape");
    }
}

/// E2E: Softmax dim=-1 resolved to last dim (dim=2 for rank-3 input).
#[test]
fn test_import_layernorm_softmax_dim() {
    let imported = import_layernorm_softmax_fixture();
    let nodes = imported.graph.nodes();

    let sm = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Softmax { .. }))
        .unwrap();
    if let TraceOp::Softmax { dim } = sm.op() {
        assert_eq!(
            *dim, 2,
            "softmax dim=-1 on rank-3 tensor should resolve to dim=2"
        );
    }
}

/// E2E: Intermediate shapes propagate correctly through LayerNorm -> Softmax.
#[test]
fn test_import_layernorm_softmax_shapes() {
    let imported = import_layernorm_softmax_fixture();
    let nodes = imported.graph.nodes();

    let input = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Input))
        .unwrap();
    assert_eq!(input.output_shape(), &[1, 4, 16]);

    let ln = nodes.iter().find(|n| n.name() == "layer_norm").unwrap();
    assert_eq!(ln.output_shape(), &[1, 4, 16]);

    let sm = nodes.iter().find(|n| n.name() == "softmax").unwrap();
    assert_eq!(sm.output_shape(), &[1, 4, 16]);
}

// ---------------------------------------------------------------------------
// GroupNorm tests
// ---------------------------------------------------------------------------

/// E2E: GroupNorm -> ReLU graph imports with correct structure.
///
/// Graph: group_norm(x[1,16,8,8], num_groups=4, w[16], b[16]) -> relu -> [1,16,8,8]
#[test]
fn test_import_groupnorm_block_structure() {
    let imported = import_groupnorm_fixture();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["relu"]);

    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::Relu),
        "expected Relu as output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 16, 8, 8]);
}

/// E2E: GroupNorm + ReLU op counts.
#[test]
fn test_import_groupnorm_block_op_counts() {
    let imported = import_groupnorm_fixture();
    let nodes = imported.graph.nodes();
    let count = |pred: fn(&TraceOp) -> bool| nodes.iter().filter(|n| pred(n.op())).count();

    assert_eq!(count(|op| matches!(op, TraceOp::GroupNorm { .. })), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::Relu)), 1);
}

/// E2E: GroupNorm parameters (num_groups, eps, weight/bias shape) survive import.
#[test]
fn test_import_groupnorm_block_params() {
    let imported = import_groupnorm_fixture();
    let nodes = imported.graph.nodes();

    let gn = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::GroupNorm { .. }))
        .unwrap();
    if let TraceOp::GroupNorm {
        num_groups,
        eps,
        weight,
        bias,
    } = gn.op()
    {
        assert_eq!(*num_groups, 4, "expected num_groups=4");
        assert!((*eps - 1e-6).abs() < 1e-10, "expected eps=1e-6, got {eps}");
        assert_eq!(weight.shape(), &[16], "GroupNorm weight shape");
        assert_eq!(bias.shape(), &[16], "GroupNorm bias shape");
    }
}

/// E2E: Intermediate shapes propagate correctly through GroupNorm -> ReLU.
#[test]
fn test_import_groupnorm_block_shapes() {
    let imported = import_groupnorm_fixture();
    let nodes = imported.graph.nodes();

    let input = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Input))
        .unwrap();
    assert_eq!(input.output_shape(), &[1, 16, 8, 8]);

    let gn = nodes.iter().find(|n| n.name() == "group_norm").unwrap();
    assert_eq!(gn.output_shape(), &[1, 16, 8, 8]);

    let relu = nodes.iter().find(|n| n.name() == "relu").unwrap();
    assert_eq!(relu.output_shape(), &[1, 16, 8, 8]);
}
