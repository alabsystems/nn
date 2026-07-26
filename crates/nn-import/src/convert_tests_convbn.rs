// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ConvBnAct backbone convert tests: Conv2d -> BatchNorm -> SiLU (standalone ops).
//!
//! Exercises the standalone `aten::conv2d.default` and `aten::batch_norm.default`
//! op mappings (as opposed to `aten::convolution.default` and
//! `_native_batch_norm_legit_no_training.default` tested in convert_tests_dpdf.rs).
//!
//! Models a 2-layer ConvBnAct backbone typical of DocLayout-YOLO / PaddleOCR DB:
//! Conv2d(3, 16, 3, stride=2, padding=1) -> BN -> SiLU ->
//! Conv2d(16, 32, 3, stride=2, padding=1) -> BN -> SiLU

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use nn_core::dyn_tensor::trace::TraceOp;

use crate::graph_build::ImportedGraph;
use crate::import_model;

static CONVBN_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Write synthetic ConvBnAct backbone weights to a safetensors file.
///
/// Stage 0: Conv2d [16, 3, 3, 3] = 432 elements, bias [16]
///          BN weight [16], bias [16], running_mean [16], running_var [16]
/// Stage 1: Conv2d [32, 16, 3, 3] = 4608 elements, bias [32]
///          BN weight [32], bias [32], running_mean [32], running_var [32]
fn write_convbnact_weights(dir: &Path) -> std::path::PathBuf {
    let mut tensors = HashMap::new();

    // Stage 0 conv: [16, 3, 3, 3] = 432 elements
    let conv1_w: Vec<u8> = (0..432)
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();
    let conv1_b: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn1_w: Vec<u8> = [1.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn1_b: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn1_mean: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn1_var: Vec<u8> = [1.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();

    // Stage 1 conv: [32, 16, 3, 3] = 4608 elements
    let conv2_w: Vec<u8> = (0..4608)
        .flat_map(|i| ((i as f32) * 0.0001).to_le_bytes())
        .collect();
    let conv2_b: Vec<u8> = [0.0f32; 32].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn2_w: Vec<u8> = [1.0f32; 32].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn2_b: Vec<u8> = [0.0f32; 32].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn2_mean: Vec<u8> = [0.0f32; 32].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn2_var: Vec<u8> = [1.0f32; 32].iter().flat_map(|f| f.to_le_bytes()).collect();

    for (name, shape, data) in [
        ("stage0.conv.weight", vec![16, 3, 3, 3], conv1_w.as_slice()),
        ("stage0.conv.bias", vec![16], conv1_b.as_slice()),
        ("stage0.bn.weight", vec![16], bn1_w.as_slice()),
        ("stage0.bn.bias", vec![16], bn1_b.as_slice()),
        ("stage0.bn.running_mean", vec![16], bn1_mean.as_slice()),
        ("stage0.bn.running_var", vec![16], bn1_var.as_slice()),
        ("stage1.conv.weight", vec![32, 16, 3, 3], conv2_w.as_slice()),
        ("stage1.conv.bias", vec![32], conv2_b.as_slice()),
        ("stage1.bn.weight", vec![32], bn2_w.as_slice()),
        ("stage1.bn.bias", vec![32], bn2_b.as_slice()),
        ("stage1.bn.running_mean", vec![32], bn2_mean.as_slice()),
        ("stage1.bn.running_var", vec![32], bn2_var.as_slice()),
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

/// Import the ConvBnAct backbone fixture from disk.
fn import_convbnact_fixture() -> ImportedGraph {
    let id = CONVBN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "nn_import_convbnact_{}_{}",
        std::process::id(),
        id
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/convbnact_backbone.json"),
    )
    .unwrap();
    let weights_path = write_convbnact_weights(&dir);
    let imported = import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    imported
}

// ---------------------------------------------------------------------------
// Graph structure tests (no Metal required)
// ---------------------------------------------------------------------------

/// E2E: ConvBnAct backbone imports with correct structure.
///
/// Exercises standalone aten::conv2d.default and aten::batch_norm.default
/// mappings (distinct from the unified convolution.default / native_batch_norm
/// paths tested in convert_tests_dpdf.rs).
#[test]
fn test_import_convbnact_backbone_structure() {
    let imported = import_convbnact_fixture();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["silu2"]);

    // 1 Input + 12 params/buffers + 6 compute ops = 19 total nodes.
    assert_eq!(imported.graph.len(), 19);

    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::Silu),
        "expected Silu as output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 32, 8, 8]);
}

/// E2E: all ConvBnAct-specific aten ops map to correct TraceOp variants.
#[test]
fn test_import_convbnact_backbone_op_counts() {
    let imported = import_convbnact_fixture();
    let nodes = imported.graph.nodes();
    let count = |pred: fn(&TraceOp) -> bool| nodes.iter().filter(|n| pred(n.op())).count();

    assert_eq!(count(|op| matches!(op, TraceOp::Conv2d { .. })), 2);
    assert_eq!(count(|op| matches!(op, TraceOp::BatchNorm { .. })), 2);
    assert_eq!(count(|op| matches!(op, TraceOp::Silu)), 2);
}

/// E2E: intermediate shapes propagate correctly through stride-2 convolutions.
#[test]
fn test_import_convbnact_backbone_shapes() {
    let imported = import_convbnact_fixture();
    let nodes = imported.graph.nodes();

    // Input: [1, 3, 32, 32]
    let input = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Input))
        .unwrap();
    assert_eq!(input.output_shape(), &[1, 3, 32, 32]);

    // Conv1 output: [1, 16, 16, 16] (stride=2, padding=1)
    let conv1 = nodes.iter().find(|n| n.name() == "conv1").unwrap();
    assert_eq!(conv1.output_shape(), &[1, 16, 16, 16]);

    // BN1 output: same shape as conv1
    let bn1 = nodes.iter().find(|n| n.name() == "bn1").unwrap();
    assert_eq!(bn1.output_shape(), &[1, 16, 16, 16]);

    // SiLU1 output: same shape
    let silu1 = nodes.iter().find(|n| n.name() == "silu1").unwrap();
    assert_eq!(silu1.output_shape(), &[1, 16, 16, 16]);

    // Conv2 output: [1, 32, 8, 8] (stride=2, padding=1)
    let conv2 = nodes.iter().find(|n| n.name() == "conv2").unwrap();
    assert_eq!(conv2.output_shape(), &[1, 32, 8, 8]);

    // BN2 + SiLU2: same shape as conv2
    let silu2 = nodes.iter().find(|n| n.name() == "silu2").unwrap();
    assert_eq!(silu2.output_shape(), &[1, 32, 8, 8]);
}

/// E2E: BatchNorm parameters (eps) survive import via standalone batch_norm path.
#[test]
fn test_import_convbnact_backbone_bn_params() {
    let imported = import_convbnact_fixture();
    let nodes = imported.graph.nodes();

    let bn_nodes: Vec<_> = nodes
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::BatchNorm { .. }))
        .collect();
    assert_eq!(bn_nodes.len(), 2);

    for bn in &bn_nodes {
        if let TraceOp::BatchNorm { eps, .. } = bn.op() {
            assert!((*eps - 1e-5).abs() < 1e-8, "expected eps=1e-5, got {eps}");
        }
    }
}

/// E2E: Conv2d parameters (stride, padding, dilation, groups) survive import
/// via standalone conv2d path.
#[test]
fn test_import_convbnact_backbone_conv_params() {
    let imported = import_convbnact_fixture();
    let nodes = imported.graph.nodes();

    let conv_nodes: Vec<_> = nodes
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Conv2d { .. }))
        .collect();
    assert_eq!(conv_nodes.len(), 2);

    for conv in &conv_nodes {
        if let TraceOp::Conv2d {
            stride,
            padding,
            dilation,
            groups,
            ..
        } = conv.op()
        {
            assert_eq!(*stride, [2, 2], "stride for {}", conv.name());
            assert_eq!(*padding, [1, 1], "padding for {}", conv.name());
            assert_eq!(*dilation, [1, 1], "dilation for {}", conv.name());
            assert_eq!(*groups, 1, "groups for {}", conv.name());
        }
    }
}
