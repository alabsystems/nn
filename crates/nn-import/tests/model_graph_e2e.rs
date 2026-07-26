// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end model graph conversion integration tests.
//!
//! Each test constructs a complete torch.export JSON graph fixture representing
//! a real model architecture pattern, writes synthetic safetensors weights,
//! runs the full import pipeline (parse -> weight load -> op map -> graph build),
//! and verifies the resulting TraceOp sequence and output shapes.
//!
//! These tests exercise the complete nn-import conversion pipeline without
//! requiring Metal, PyTorch, or external model files.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;
use nn_import::{build_graph, build_weight_map, parse_exported_program};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write synthetic safetensors weights to a temp dir and return the parsed
/// ExportedProgram along with a weight map.
fn import_fixture(
    json: &str,
    raw_weights: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> nn_import::ImportedGraph {
    let program = parse_exported_program(json.as_bytes())
        .expect("fixture JSON must parse as valid ExportedProgram");
    let weight_map = build_weight_map(&program.graph_module.signature.input_specs, &raw_weights);
    build_graph(&program, &weight_map).expect("build_graph must succeed for fixture")
}

/// Collect only the compute ops (non-Input, non-Constant) from an imported graph.
fn compute_ops(
    imported: &nn_import::ImportedGraph,
) -> Vec<&nn_core::dyn_tensor::trace::TraceNode> {
    imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .collect()
}

// ---------------------------------------------------------------------------
// (a) ResNet basic block: Conv2d -> BN -> ReLU -> Conv2d -> BN + skip -> ReLU
// ---------------------------------------------------------------------------

fn resnet_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    // conv1: [16, 16, 3, 3] = 2304 elements
    w.insert(
        "conv1.weight".to_string(),
        (vec![0.01; 2304], vec![16, 16, 3, 3]),
    );
    w.insert("conv1.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert("bn1.weight".to_string(), (vec![1.0; 16], vec![16]));
    w.insert("bn1.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert("bn1.running_mean".to_string(), (vec![0.0; 16], vec![16]));
    w.insert("bn1.running_var".to_string(), (vec![1.0; 16], vec![16]));
    // conv2: [16, 16, 3, 3] = 2304 elements
    w.insert(
        "conv2.weight".to_string(),
        (vec![0.01; 2304], vec![16, 16, 3, 3]),
    );
    w.insert("conv2.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert("bn2.weight".to_string(), (vec![1.0; 16], vec![16]));
    w.insert("bn2.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert("bn2.running_mean".to_string(), (vec![0.0; 16], vec![16]));
    w.insert("bn2.running_var".to_string(), (vec![1.0; 16], vec![16]));
    w
}

#[test]
fn test_resnet_basic_block_op_sequence() {
    let imported = import_fixture(
        include_str!("../test_data/resnet_basic_block.json"),
        resnet_weights(),
    );
    let ops = compute_ops(&imported);

    // Expected: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> Add -> ReLU
    assert_eq!(ops.len(), 7, "ResNet basic block has 7 compute ops");
    assert!(
        matches!(ops[0].op(), TraceOp::Conv2d { .. }),
        "op[0] = Conv2d, got {:?}",
        ops[0].op()
    );
    assert!(
        matches!(ops[1].op(), TraceOp::BatchNorm { .. }),
        "op[1] = BatchNorm, got {:?}",
        ops[1].op()
    );
    assert!(
        matches!(ops[2].op(), TraceOp::Relu),
        "op[2] = ReLU, got {:?}",
        ops[2].op()
    );
    assert!(
        matches!(ops[3].op(), TraceOp::Conv2d { .. }),
        "op[3] = Conv2d, got {:?}",
        ops[3].op()
    );
    assert!(
        matches!(ops[4].op(), TraceOp::BatchNorm { .. }),
        "op[4] = BatchNorm, got {:?}",
        ops[4].op()
    );
    assert!(
        matches!(ops[5].op(), TraceOp::Add),
        "op[5] = Add (skip), got {:?}",
        ops[5].op()
    );
    assert!(
        matches!(ops[6].op(), TraceOp::Relu),
        "op[6] = ReLU, got {:?}",
        ops[6].op()
    );
}

#[test]
fn test_resnet_basic_block_shapes() {
    let imported = import_fixture(
        include_str!("../test_data/resnet_basic_block.json"),
        resnet_weights(),
    );
    let ops = compute_ops(&imported);

    // All ops preserve spatial dims: [1, 16, 8, 8] throughout (same-padding convs).
    let expected_shape: &[usize] = &[1, 16, 8, 8];
    for (i, op) in ops.iter().enumerate() {
        assert_eq!(
            op.output_shape(),
            expected_shape,
            "ResNet op[{i}] shape mismatch: expected {expected_shape:?}, got {:?}",
            op.output_shape()
        );
    }
}

#[test]
fn test_resnet_basic_block_skip_connection() {
    let imported = import_fixture(
        include_str!("../test_data/resnet_basic_block.json"),
        resnet_weights(),
    );

    // The Add node should have 2 inputs (bn2 output + original input).
    let add_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Add))
        .expect("must have Add node for skip connection");
    assert_eq!(add_node.inputs().len(), 2, "skip Add must have 2 inputs");
}

#[test]
fn test_resnet_basic_block_metadata() {
    let imported = import_fixture(
        include_str!("../test_data/resnet_basic_block.json"),
        resnet_weights(),
    );
    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["relu_out"]);
    // Output is same shape as input (identity residual block).
    let output = imported.graph.output_node().unwrap();
    assert_eq!(output.output_shape(), &[1, 16, 8, 8]);
}

// ---------------------------------------------------------------------------
// (b) Transformer encoder layer:
//     LN -> Linear(Q,K,V) -> SDPA -> Linear(O) -> residual ->
//     LN -> Linear -> GELU -> Linear -> residual
// ---------------------------------------------------------------------------

fn transformer_encoder_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    // d_model = 16, n_heads = 2, d_head = 8, d_ff = 32
    w.insert("ln1.weight".to_string(), (vec![1.0; 16], vec![16]));
    w.insert("ln1.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert(
        "attn.q_proj.weight".to_string(),
        (vec![0.01; 256], vec![16, 16]),
    );
    w.insert("attn.q_proj.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert(
        "attn.k_proj.weight".to_string(),
        (vec![0.01; 256], vec![16, 16]),
    );
    w.insert("attn.k_proj.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert(
        "attn.v_proj.weight".to_string(),
        (vec![0.01; 256], vec![16, 16]),
    );
    w.insert("attn.v_proj.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert(
        "attn.out_proj.weight".to_string(),
        (vec![0.01; 256], vec![16, 16]),
    );
    w.insert("attn.out_proj.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert("ln2.weight".to_string(), (vec![1.0; 16], vec![16]));
    w.insert("ln2.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert("ff.fc1.weight".to_string(), (vec![0.01; 512], vec![32, 16]));
    w.insert("ff.fc1.bias".to_string(), (vec![0.0; 32], vec![32]));
    w.insert("ff.fc2.weight".to_string(), (vec![0.01; 512], vec![16, 32]));
    w.insert("ff.fc2.bias".to_string(), (vec![0.0; 16], vec![16]));
    w
}

#[test]
fn test_transformer_encoder_op_sequence() {
    let imported = import_fixture(
        include_str!("../test_data/transformer_encoder_layer.json"),
        transformer_encoder_weights(),
    );
    let ops = compute_ops(&imported);

    // Expected sequence:
    // LN1, Q_proj, K_proj, V_proj, Reshape x3, Transpose x3, SDPA,
    // Transpose, Reshape, O_proj, Add(residual1),
    // LN2, FF1, GELU, FF2, Add(residual2)
    assert_eq!(
        ops.len(),
        20,
        "Transformer encoder layer has 20 compute ops, got {}",
        ops.len()
    );

    // Verify key ops are present.
    let ln_count = ops
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::LayerNorm { .. }))
        .count();
    assert_eq!(ln_count, 2, "expected 2 LayerNorm ops");

    let linear_count = ops
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Linear { .. }))
        .count();
    assert_eq!(linear_count, 6, "expected 6 Linear ops (Q,K,V,O,FF1,FF2)");

    let sdpa_count = ops
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Sdpa { .. }))
        .count();
    assert_eq!(sdpa_count, 1, "expected 1 SDPA op");

    let gelu_count = ops
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Gelu | TraceOp::GeluErf))
        .count();
    assert_eq!(gelu_count, 1, "expected 1 GELU/GeluErf op");

    let add_count = ops
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Add))
        .count();
    assert_eq!(add_count, 2, "expected 2 Add ops (residual connections)");

    let reshape_count = ops
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Reshape { .. }))
        .count();
    assert_eq!(reshape_count, 4, "expected 4 Reshape ops");

    let transpose_count = ops
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Transpose { .. }))
        .count();
    assert_eq!(transpose_count, 4, "expected 4 Transpose ops");
}

#[test]
fn test_transformer_encoder_output_shape() {
    let imported = import_fixture(
        include_str!("../test_data/transformer_encoder_layer.json"),
        transformer_encoder_weights(),
    );

    // Output shape must match input shape (residual preserves dimensions).
    let output = imported.graph.output_node().unwrap();
    assert_eq!(
        output.output_shape(),
        &[1, 4, 16],
        "encoder output = [B, seq_len, d_model]"
    );
    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.output_names, vec!["residual2"]);
}

#[test]
fn test_transformer_encoder_residual_connections() {
    let imported = import_fixture(
        include_str!("../test_data/transformer_encoder_layer.json"),
        transformer_encoder_weights(),
    );
    let ops = compute_ops(&imported);

    // Both Add nodes should have exactly 2 inputs.
    let adds: Vec<_> = ops
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Add))
        .collect();
    assert_eq!(adds.len(), 2);
    for add in &adds {
        assert_eq!(add.inputs().len(), 2, "residual Add must have 2 inputs");
    }
}

// ---------------------------------------------------------------------------
// (c) Embedding + positional: two embeddings summed
// ---------------------------------------------------------------------------

fn embedding_positional_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    // Token embedding: vocab=100, dim=16 => 1600 elements
    w.insert(
        "tok_embed.weight".to_string(),
        (vec![0.01; 1600], vec![100, 16]),
    );
    // Positional embedding: max_pos=32, dim=16 => 512 elements
    w.insert(
        "pos_embed.weight".to_string(),
        (vec![0.01; 512], vec![32, 16]),
    );
    w
}

#[test]
fn test_embedding_positional_op_sequence() {
    let imported = import_fixture(
        include_str!("../test_data/embedding_positional.json"),
        embedding_positional_weights(),
    );
    let ops = compute_ops(&imported);

    // Expected: Embedding(tok) -> Embedding(pos) -> Add
    assert_eq!(ops.len(), 3, "Embedding+positional has 3 compute ops");
    assert!(
        matches!(ops[0].op(), TraceOp::Embedding { .. }),
        "op[0] = Embedding, got {:?}",
        ops[0].op()
    );
    assert!(
        matches!(ops[1].op(), TraceOp::Embedding { .. }),
        "op[1] = Embedding, got {:?}",
        ops[1].op()
    );
    assert!(
        matches!(ops[2].op(), TraceOp::Add),
        "op[2] = Add, got {:?}",
        ops[2].op()
    );
}

#[test]
fn test_embedding_positional_shapes() {
    let imported = import_fixture(
        include_str!("../test_data/embedding_positional.json"),
        embedding_positional_weights(),
    );
    let ops = compute_ops(&imported);

    // Both embeddings produce [1, 8, 16], Add produces [1, 8, 16].
    assert_eq!(ops[0].output_shape(), &[1, 8, 16]);
    assert_eq!(ops[1].output_shape(), &[1, 8, 16]);
    assert_eq!(ops[2].output_shape(), &[1, 8, 16]);
}

#[test]
fn test_embedding_positional_multi_input() {
    let imported = import_fixture(
        include_str!("../test_data/embedding_positional.json"),
        embedding_positional_weights(),
    );
    // This model has 2 user inputs: token_ids and pos_ids.
    assert_eq!(imported.num_user_inputs, 2);
    assert_eq!(imported.user_input_names, vec!["token_ids", "pos_ids"]);
}

// ---------------------------------------------------------------------------
// (d) Classification head: Linear -> Softmax
// ---------------------------------------------------------------------------

fn classification_head_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    // fc: 64 -> 10 = 640 elements
    w.insert("fc.weight".to_string(), (vec![0.01; 640], vec![10, 64]));
    w.insert("fc.bias".to_string(), (vec![0.0; 10], vec![10]));
    w
}

#[test]
fn test_classification_head_op_sequence() {
    let imported = import_fixture(
        include_str!("../test_data/classification_head.json"),
        classification_head_weights(),
    );
    let ops = compute_ops(&imported);

    assert_eq!(ops.len(), 2, "Classification head has 2 compute ops");
    assert!(
        matches!(ops[0].op(), TraceOp::Linear { .. }),
        "op[0] = Linear, got {:?}",
        ops[0].op()
    );
    assert!(
        matches!(ops[1].op(), TraceOp::Softmax { .. }),
        "op[1] = Softmax, got {:?}",
        ops[1].op()
    );
}

#[test]
fn test_classification_head_shapes() {
    let imported = import_fixture(
        include_str!("../test_data/classification_head.json"),
        classification_head_weights(),
    );
    let ops = compute_ops(&imported);

    // Linear: [1, 64] -> [1, 10]
    assert_eq!(ops[0].output_shape(), &[1, 10]);
    // Softmax preserves shape: [1, 10]
    assert_eq!(ops[1].output_shape(), &[1, 10]);
}

#[test]
fn test_classification_head_softmax_dim() {
    let imported = import_fixture(
        include_str!("../test_data/classification_head.json"),
        classification_head_weights(),
    );
    let ops = compute_ops(&imported);

    // Softmax dim should resolve to the last axis.
    if let TraceOp::Softmax { dim } = ops[1].op() {
        // dim=-1 on a 2D tensor resolves to dim=1.
        assert_eq!(*dim, 1, "Softmax should be along dim=1 (last axis)");
    } else {
        panic!("expected Softmax op");
    }
}

// ---------------------------------------------------------------------------
// (e) Detection head: Linear -> Sigmoid (bbox) + Linear -> Softmax (cls)
// ---------------------------------------------------------------------------

fn detection_head_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    // bbox: 32 -> 4 = 128 elements
    w.insert(
        "bbox_head.weight".to_string(),
        (vec![0.01; 128], vec![4, 32]),
    );
    w.insert("bbox_head.bias".to_string(), (vec![0.0; 4], vec![4]));
    // cls: 32 -> 5 = 160 elements
    w.insert(
        "cls_head.weight".to_string(),
        (vec![0.01; 160], vec![5, 32]),
    );
    w.insert("cls_head.bias".to_string(), (vec![0.0; 5], vec![5]));
    w
}

#[test]
fn test_detection_head_op_sequence() {
    let imported = import_fixture(
        include_str!("../test_data/detection_head.json"),
        detection_head_weights(),
    );
    let ops = compute_ops(&imported);

    // Expected: Linear(bbox) -> Sigmoid -> Linear(cls) -> Softmax
    assert_eq!(ops.len(), 4, "Detection head has 4 compute ops");
    assert!(
        matches!(ops[0].op(), TraceOp::Linear { .. }),
        "op[0] = Linear(bbox), got {:?}",
        ops[0].op()
    );
    assert!(
        matches!(ops[1].op(), TraceOp::Sigmoid),
        "op[1] = Sigmoid, got {:?}",
        ops[1].op()
    );
    assert!(
        matches!(ops[2].op(), TraceOp::Linear { .. }),
        "op[2] = Linear(cls), got {:?}",
        ops[2].op()
    );
    assert!(
        matches!(ops[3].op(), TraceOp::Softmax { .. }),
        "op[3] = Softmax, got {:?}",
        ops[3].op()
    );
}

#[test]
fn test_detection_head_shapes() {
    let imported = import_fixture(
        include_str!("../test_data/detection_head.json"),
        detection_head_weights(),
    );
    let ops = compute_ops(&imported);

    // bbox path: [1, 32] -> Linear -> [1, 4] -> Sigmoid -> [1, 4]
    assert_eq!(ops[0].output_shape(), &[1, 4]);
    assert_eq!(ops[1].output_shape(), &[1, 4]);
    // cls path: [1, 32] -> Linear -> [1, 5] -> Softmax -> [1, 5]
    assert_eq!(ops[2].output_shape(), &[1, 5]);
    assert_eq!(ops[3].output_shape(), &[1, 5]);
}

#[test]
fn test_detection_head_dual_branch_from_same_input() {
    let imported = import_fixture(
        include_str!("../test_data/detection_head.json"),
        detection_head_weights(),
    );
    let ops = compute_ops(&imported);

    // Both Linear ops should have the same first input (the shared features tensor).
    let bbox_linear_inputs = ops[0].inputs();
    let cls_linear_inputs = ops[2].inputs();
    // First input of both Linears should be the user input node.
    assert_eq!(
        bbox_linear_inputs[0], cls_linear_inputs[0],
        "Both branches should share the same input features tensor"
    );
}

// ---------------------------------------------------------------------------
// (f) Conv backbone with stride downsampling:
//     Conv2d(stride=2) -> BN -> SiLU -> Conv2d(stride=2) -> BN -> SiLU
//     (Uses existing convbnact_backbone.json fixture)
// ---------------------------------------------------------------------------

fn conv_backbone_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    // conv1: [16, 3, 3, 3] = 432 elements
    w.insert(
        "stage0.conv.weight".to_string(),
        (vec![0.01; 432], vec![16, 3, 3, 3]),
    );
    w.insert("stage0.conv.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert("stage0.bn.weight".to_string(), (vec![1.0; 16], vec![16]));
    w.insert("stage0.bn.bias".to_string(), (vec![0.0; 16], vec![16]));
    w.insert(
        "stage0.bn.running_mean".to_string(),
        (vec![0.0; 16], vec![16]),
    );
    w.insert(
        "stage0.bn.running_var".to_string(),
        (vec![1.0; 16], vec![16]),
    );
    // conv2: [32, 16, 3, 3] = 4608 elements
    w.insert(
        "stage1.conv.weight".to_string(),
        (vec![0.01; 4608], vec![32, 16, 3, 3]),
    );
    w.insert("stage1.conv.bias".to_string(), (vec![0.0; 32], vec![32]));
    w.insert("stage1.bn.weight".to_string(), (vec![1.0; 32], vec![32]));
    w.insert("stage1.bn.bias".to_string(), (vec![0.0; 32], vec![32]));
    w.insert(
        "stage1.bn.running_mean".to_string(),
        (vec![0.0; 32], vec![32]),
    );
    w.insert(
        "stage1.bn.running_var".to_string(),
        (vec![1.0; 32], vec![32]),
    );
    w
}

#[test]
fn test_conv_backbone_op_sequence() {
    let imported = import_fixture(
        include_str!("../test_data/convbnact_backbone.json"),
        conv_backbone_weights(),
    );
    let ops = compute_ops(&imported);

    // Expected: Conv2d -> BatchNorm -> SiLU -> Conv2d -> BatchNorm -> SiLU
    assert_eq!(
        ops.len(),
        6,
        "Conv backbone has 6 compute ops, got {}",
        ops.len()
    );
    assert!(
        matches!(ops[0].op(), TraceOp::Conv2d { .. }),
        "op[0] = Conv2d, got {:?}",
        ops[0].op()
    );
    assert!(
        matches!(ops[1].op(), TraceOp::BatchNorm { .. }),
        "op[1] = BatchNorm, got {:?}",
        ops[1].op()
    );
    assert!(
        matches!(ops[2].op(), TraceOp::Silu),
        "op[2] = SiLU, got {:?}",
        ops[2].op()
    );
    assert!(
        matches!(ops[3].op(), TraceOp::Conv2d { .. }),
        "op[3] = Conv2d, got {:?}",
        ops[3].op()
    );
    assert!(
        matches!(ops[4].op(), TraceOp::BatchNorm { .. }),
        "op[4] = BatchNorm, got {:?}",
        ops[4].op()
    );
    assert!(
        matches!(ops[5].op(), TraceOp::Silu),
        "op[5] = SiLU, got {:?}",
        ops[5].op()
    );
}

#[test]
fn test_conv_backbone_spatial_downsampling() {
    let imported = import_fixture(
        include_str!("../test_data/convbnact_backbone.json"),
        conv_backbone_weights(),
    );
    let ops = compute_ops(&imported);

    // Input: [1, 3, 32, 32]
    // After Conv1 (pad=1, stride=2): [1, 16, 16, 16]
    assert_eq!(ops[0].output_shape(), &[1, 16, 16, 16]);
    // After BN1: [1, 16, 16, 16]
    assert_eq!(ops[1].output_shape(), &[1, 16, 16, 16]);
    // After SiLU: [1, 16, 16, 16]
    assert_eq!(ops[2].output_shape(), &[1, 16, 16, 16]);
    // After Conv2 (pad=1, stride=2): [1, 32, 8, 8]
    assert_eq!(ops[3].output_shape(), &[1, 32, 8, 8]);
    // After BN2: [1, 32, 8, 8]
    assert_eq!(ops[4].output_shape(), &[1, 32, 8, 8]);
    // After SiLU: [1, 32, 8, 8]
    assert_eq!(ops[5].output_shape(), &[1, 32, 8, 8]);
}

// ---------------------------------------------------------------------------
// (g) Multi-input graph: two inputs merged via cat -> ReLU -> Linear
// ---------------------------------------------------------------------------

fn multi_input_weights() -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    let mut w = HashMap::new();
    // fc: 16 -> 4 = 64 elements
    w.insert("fc.weight".to_string(), (vec![0.01; 64], vec![4, 16]));
    w.insert("fc.bias".to_string(), (vec![0.0; 4], vec![4]));
    w
}

#[test]
fn test_multi_input_cat_op_sequence() {
    let imported = import_fixture(
        include_str!("../test_data/multi_input_cat.json"),
        multi_input_weights(),
    );
    let ops = compute_ops(&imported);

    // Expected: Cat -> ReLU -> Linear
    assert_eq!(ops.len(), 3, "Multi-input graph has 3 compute ops");
    assert!(
        matches!(ops[0].op(), TraceOp::Cat { dim: 1, .. }),
        "op[0] = Cat(dim=1), got {:?}",
        ops[0].op()
    );
    assert!(
        matches!(ops[1].op(), TraceOp::Relu),
        "op[1] = ReLU, got {:?}",
        ops[1].op()
    );
    assert!(
        matches!(ops[2].op(), TraceOp::Linear { .. }),
        "op[2] = Linear, got {:?}",
        ops[2].op()
    );
}

#[test]
fn test_multi_input_cat_shapes() {
    let imported = import_fixture(
        include_str!("../test_data/multi_input_cat.json"),
        multi_input_weights(),
    );
    let ops = compute_ops(&imported);

    // Cat([1,8], [1,8], dim=1) -> [1, 16]
    assert_eq!(ops[0].output_shape(), &[1, 16]);
    // ReLU preserves shape.
    assert_eq!(ops[1].output_shape(), &[1, 16]);
    // Linear: [1, 16] -> [1, 4]
    assert_eq!(ops[2].output_shape(), &[1, 4]);
}

#[test]
fn test_multi_input_two_user_inputs() {
    let imported = import_fixture(
        include_str!("../test_data/multi_input_cat.json"),
        multi_input_weights(),
    );
    assert_eq!(imported.num_user_inputs, 2);
    assert_eq!(imported.user_input_names, vec!["a", "b"]);
}

#[test]
fn test_multi_input_cat_num_inputs() {
    let imported = import_fixture(
        include_str!("../test_data/multi_input_cat.json"),
        multi_input_weights(),
    );
    let ops = compute_ops(&imported);

    // Cat node should reference both user inputs.
    if let TraceOp::Cat { dim, num_inputs } = ops[0].op() {
        assert_eq!(*dim, 1);
        assert_eq!(*num_inputs, 2);
    } else {
        panic!("expected Cat op");
    }
}

// ---------------------------------------------------------------------------
// Topology validation: every node's inputs exist in the graph
// ---------------------------------------------------------------------------

#[test]
fn test_all_fixtures_topology_valid() {
    let fixtures: Vec<(&str, HashMap<String, (Vec<f32>, Vec<usize>)>)> = vec![
        (
            include_str!("../test_data/resnet_basic_block.json"),
            resnet_weights(),
        ),
        (
            include_str!("../test_data/transformer_encoder_layer.json"),
            transformer_encoder_weights(),
        ),
        (
            include_str!("../test_data/embedding_positional.json"),
            embedding_positional_weights(),
        ),
        (
            include_str!("../test_data/classification_head.json"),
            classification_head_weights(),
        ),
        (
            include_str!("../test_data/detection_head.json"),
            detection_head_weights(),
        ),
        (
            include_str!("../test_data/convbnact_backbone.json"),
            conv_backbone_weights(),
        ),
        (
            include_str!("../test_data/multi_input_cat.json"),
            multi_input_weights(),
        ),
    ];

    for (i, (json, weights)) in fixtures.iter().enumerate() {
        let imported = import_fixture(json, weights.clone());
        for node in imported.graph.nodes() {
            for &input_id in node.inputs() {
                assert!(
                    imported.graph.node(input_id).is_some(),
                    "fixture[{i}]: node '{}' references missing input_id {}",
                    node.name(),
                    input_id
                );
            }
        }
    }
}
