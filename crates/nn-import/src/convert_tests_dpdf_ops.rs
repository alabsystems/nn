// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! dpdf extended op coverage tests — validates all ops needed by the 6 dpdf
//! model architectures are supported by the nn::convert() graph builder.
//!
//! Architecture coverage:
//! - DocLayout-YOLO: Conv2d, BN, SiLU, MaxPool2d, Upsample2d, Cat, HardSwish
//! - Granite-Docling: Embedding, LayerNorm, GELU, MatMul, Softmax, attention
//! - PaddleOCR-VL: Conv2d, BN, ReLU, attention, LSTM (base + ext)
//! - Table Transformer: Conv2d, LayerNorm, attention, FFN, upsample bilinear
//! - Qwen3-VL: RMSNorm, RoPE (decomposed), GQA (decomposed), SwiGLU (decomposed)
//! - FireRed-OCR: Conv2d, BN, CTC decoder (argmax)
//!
//! Part of dpdf model import (Wave 5 TL1).

use std::collections::HashMap;
use std::path::Path;

use nn_core::dyn_tensor::trace::TraceOp;

use crate::op_map::{map_node_to_trace_op, OpMapContext, ResolvedWeight};
use crate::parse::{
    Argument, ArgumentFloat, ArgumentInt, ArgumentInts, ArgumentTensor, NamedArgument, Node,
    TensorArgument, TensorMeta,
};

// ---------------------------------------------------------------------------
// Helper: build a minimal node for unit testing individual op mappers
// ---------------------------------------------------------------------------

fn make_node(target: &str, inputs: Vec<NamedArgument>) -> Node {
    Node {
        target: target.to_string(),
        inputs,
        outputs: vec![],
        metadata: HashMap::default(),
    }
}

fn tensor_arg(name: &str, tensor_name: &str) -> NamedArgument {
    NamedArgument {
        name: name.to_string(),
        arg: Argument::Tensor(ArgumentTensor {
            as_tensor: TensorArgument {
                name: tensor_name.to_string(),
            },
        }),
        kind: Some(1),
    }
}

fn int_arg(name: &str, val: i64) -> NamedArgument {
    NamedArgument {
        name: name.to_string(),
        arg: Argument::Int(ArgumentInt { as_int: val }),
        kind: Some(1),
    }
}

fn float_arg(name: &str, val: f64) -> NamedArgument {
    NamedArgument {
        name: name.to_string(),
        arg: Argument::Float(ArgumentFloat { as_float: val }),
        kind: Some(1),
    }
}

fn ints_arg(name: &str, vals: &[i64]) -> NamedArgument {
    NamedArgument {
        name: name.to_string(),
        arg: Argument::Ints(ArgumentInts {
            as_ints: vals.to_vec(),
        }),
        kind: Some(1),
    }
}

fn empty_ctx() -> OpMapContext<'static> {
    // Leak the maps so we get 'static — fine for tests.
    let tensor_meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::new(HashMap::new()));
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::new(HashMap::new()));
    OpMapContext {
        tensor_meta,
        weights,
    }
}

fn ctx_with_weight(name: &str, data: Vec<f32>, shape: Vec<usize>) -> OpMapContext<'static> {
    let tensor_meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::new(HashMap::new()));
    let mut weights_map = HashMap::new();
    weights_map.insert(name.to_string(), ResolvedWeight::new(data, shape));
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::new(weights_map));
    OpMapContext {
        tensor_meta,
        weights,
    }
}

// ---------------------------------------------------------------------------
// Unit tests: individual dpdf op mappers
// ---------------------------------------------------------------------------

// -- Upsampling 2D --

#[test]
fn test_map_upsample_nearest2d() {
    let node = make_node(
        "torch.ops.aten.upsample_nearest2d.default",
        vec![
            tensor_arg("self", "x"),
            ints_arg("output_size", &[16, 16]),
            float_arg("scales_h", 2.0),
            float_arg("scales_w", 2.0),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(op, TraceOp::Upsample2d { .. }),
        "expected Upsample2d, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_upsample_bilinear2d() {
    let node = make_node(
        "torch.ops.aten.upsample_bilinear2d.default",
        vec![
            tensor_arg("self", "x"),
            ints_arg("output_size", &[16, 16]),
            float_arg("scales_h", 2.0),
            float_arg("scales_w", 2.0),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(op, TraceOp::Upsample2d { .. }),
        "expected Upsample2d, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// -- Activation --
// hardswish, hardsigmoid, mish, softplus, selu tests are in
// convert_tests_activation.rs (Wave 10). No duplication needed.

// -- Mask ops --

#[test]
fn test_map_triu() {
    let node = make_node(
        "torch.ops.aten.triu.default",
        vec![tensor_arg("self", "x"), int_arg("diagonal", 1)],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Triu { diagonal: 1 }),
        "expected Triu {{ diagonal: 1 }}, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_triu_default_diagonal() {
    let node = make_node("torch.ops.aten.triu.default", vec![tensor_arg("self", "x")]);
    let ctx = empty_ctx();
    let (op, _) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Triu { diagonal: 0 }),
        "expected diagonal=0 default, got {op:?}"
    );
}

#[test]
fn test_map_tril() {
    let node = make_node(
        "torch.ops.aten.tril.default",
        vec![tensor_arg("self", "x"), int_arg("diagonal", 0)],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Tril { diagonal: 0 }),
        "expected Tril, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// -- Selection / Indexing --

#[test]
fn test_map_gather() {
    let node = make_node(
        "torch.ops.aten.gather.default",
        vec![
            tensor_arg("self", "x"),
            int_arg("dim", 1),
            tensor_arg("index", "idx"),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Gather { dim: 1 }),
        "expected Gather {{ dim: 1 }}, got {op:?}"
    );
    assert_eq!(inputs, vec!["x", "idx"]);
}

#[test]
fn test_map_argmax() {
    let node = make_node(
        "torch.ops.aten.argmax.default",
        vec![tensor_arg("self", "x"), int_arg("dim", 2)],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Argmax { dim: 2 }),
        "expected Argmax {{ dim: 2 }}, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_argmin() {
    let node = make_node(
        "torch.ops.aten.argmin.default",
        vec![tensor_arg("self", "x"), int_arg("dim", 0)],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Argmin { dim: 0 }),
        "expected Argmin {{ dim: 0 }}, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// -- Vision --

#[test]
fn test_map_pixel_shuffle() {
    let node = make_node(
        "torch.ops.aten.pixel_shuffle.default",
        vec![tensor_arg("self", "x"), int_arg("upscale_factor", 2)],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::PixelShuffle { upscale_factor: 2 }),
        "expected PixelShuffle {{ upscale_factor: 2 }}, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_pixel_unshuffle() {
    let node = make_node(
        "torch.ops.aten.pixel_unshuffle.default",
        vec![tensor_arg("self", "x"), int_arg("downscale_factor", 2)],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::PixelUnshuffle {
                downscale_factor: 2
            }
        ),
        "expected PixelUnshuffle, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// -- RMSNorm --

#[test]
fn test_map_rms_norm() {
    let ctx = ctx_with_weight("w_rms", vec![1.0; 64], vec![64]);
    let node = make_node(
        "torch.ops.aten.rms_norm.default",
        vec![
            tensor_arg("input", "x"),
            ints_arg("normalized_shape", &[64]),
            tensor_arg("weight", "w_rms"),
            float_arg("eps", 1e-6),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 3).unwrap();
    assert!(
        matches!(op, TraceOp::RmsNorm { .. }),
        "expected RmsNorm, got {op:?}"
    );
    if let TraceOp::RmsNorm { eps, .. } = &op {
        assert!((*eps - 1e-6).abs() < 1e-10, "eps mismatch: {eps}");
    }
    assert_eq!(inputs, vec!["x"]);
}

// -- Repeat --

#[test]
fn test_map_repeat() {
    let node = make_node(
        "torch.ops.aten.repeat.default",
        vec![tensor_arg("self", "x"), ints_arg("repeats", &[1, 2, 1])],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Expand { .. }),
        "expected Expand (from repeat), got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// ---------------------------------------------------------------------------
// E2E: YOLO neck fixture (Conv2d + BN + HardSwish + MaxPool2d + Upsample2d + Cat)
// ---------------------------------------------------------------------------

fn write_yolo_neck_weights(dir: &Path) -> std::path::PathBuf {
    let mut tensors = HashMap::new();

    let conv_w: Vec<u8> = (0..432)
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();
    let conv_b: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn_w: Vec<u8> = [1.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn_b: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn_mean: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();
    let bn_var: Vec<u8> = [1.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();

    for (name, shape, data) in [
        ("conv.weight", vec![16, 3, 3, 3], conv_w.as_slice()),
        ("conv.bias", vec![16], conv_b.as_slice()),
        ("bn.weight", vec![16], bn_w.as_slice()),
        ("bn.bias", vec![16], bn_b.as_slice()),
        ("bn.running_mean", vec![16], bn_mean.as_slice()),
        ("bn.running_var", vec![16], bn_var.as_slice()),
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

#[test]
fn test_import_dpdf_yolo_neck_structure() {
    let dir = std::env::temp_dir().join(format!("nn_dpdf_yolo_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/dpdf_yolo_neck_mini.json"),
    )
    .unwrap();
    let weights_path = write_yolo_neck_weights(&dir);
    let imported = crate::import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["cat_out"]);

    let nodes = imported.graph.nodes();
    let count = |pred: fn(&TraceOp) -> bool| nodes.iter().filter(|n| pred(n.op())).count();

    assert_eq!(count(|op| matches!(op, TraceOp::Conv2d { .. })), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::BatchNorm { .. })), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::HardSwish)), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::MaxPool2d { .. })), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::Upsample2d { .. })), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::Cat { .. })), 1);

    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(
            output.op(),
            TraceOp::Cat {
                dim: 1,
                num_inputs: 2
            }
        ),
        "expected Cat as output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 32, 8, 8]);
}

// ---------------------------------------------------------------------------
// E2E: Transformer encoder fixture (Embedding + LayerNorm + GELU + Softmax + MatMul)
// ---------------------------------------------------------------------------

fn write_transformer_encoder_weights(dir: &Path) -> std::path::PathBuf {
    let mut tensors = HashMap::new();

    let emb_w: Vec<u8> = (0..6400)
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();
    let ln_w: Vec<u8> = [1.0f32; 64].iter().flat_map(|f| f.to_le_bytes()).collect();
    let ln_b: Vec<u8> = [0.0f32; 64].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc1_w: Vec<u8> = (0..4096)
        .flat_map(|i| ((i as f32) * 0.0001).to_le_bytes())
        .collect();
    let fc1_b: Vec<u8> = [0.0f32; 64].iter().flat_map(|f| f.to_le_bytes()).collect();
    let fc2_w: Vec<u8> = (0..2048)
        .flat_map(|i| ((i as f32) * 0.0001).to_le_bytes())
        .collect();
    let fc2_b: Vec<u8> = [0.0f32; 32].iter().flat_map(|f| f.to_le_bytes()).collect();

    for (name, shape, data) in [
        ("emb.weight", vec![100, 64], emb_w.as_slice()),
        ("ln.weight", vec![64], ln_w.as_slice()),
        ("ln.bias", vec![64], ln_b.as_slice()),
        ("fc1.weight", vec![64, 64], fc1_w.as_slice()),
        ("fc1.bias", vec![64], fc1_b.as_slice()),
        ("fc2.weight", vec![32, 64], fc2_w.as_slice()),
        ("fc2.bias", vec![32], fc2_b.as_slice()),
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

#[test]
fn test_import_dpdf_transformer_encoder_structure() {
    let dir = std::env::temp_dir().join(format!("nn_dpdf_xfmr_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/dpdf_transformer_encoder_mini.json"),
    )
    .unwrap();
    let weights_path = write_transformer_encoder_weights(&dir);
    let imported = crate::import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["fc2"]);

    let nodes = imported.graph.nodes();
    let count = |pred: fn(&TraceOp) -> bool| nodes.iter().filter(|n| pred(n.op())).count();

    // Embedding + LayerNorm + 2x Linear + GeluErf + Softmax + MatMul + Add
    assert_eq!(count(|op| matches!(op, TraceOp::Embedding { .. })), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::LayerNorm { .. })), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::Linear { .. })), 2);
    assert_eq!(count(|op| matches!(op, TraceOp::GeluErf)), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::Softmax { .. })), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::MatMul)), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::Add)), 1);
}

// -- Narrow --

#[test]
fn test_map_narrow() {
    let node = make_node(
        "torch.ops.aten.narrow.default",
        vec![
            tensor_arg("self", "x"),
            int_arg("dim", 1),
            int_arg("start", 2),
            int_arg("length", 5),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::Narrow {
                dim: 1,
                start: 2,
                length: 5
            }
        ),
        "expected Narrow {{ dim: 1, start: 2, length: 5 }}, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// -- TopK --

#[test]
fn test_map_topk() {
    let node = make_node(
        "torch.ops.aten.topk.default",
        vec![tensor_arg("self", "x"), int_arg("k", 5), int_arg("dim", 1)],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::Topk { k: 5, dim: 1 }),
        "expected Topk {{ k: 5, dim: 1 }}, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// -- Sort --

#[test]
fn test_map_sort() {
    let node = make_node(
        "torch.ops.aten.sort.default",
        vec![tensor_arg("self", "x"), int_arg("dim", 2)],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::Sort {
                dim: 2,
                descending: false
            }
        ),
        "expected Sort {{ dim: 2, descending: false }}, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// -- Scatter add --

#[test]
fn test_map_scatter_add() {
    let node = make_node(
        "torch.ops.aten.scatter_add.default",
        vec![
            tensor_arg("self", "x"),
            int_arg("dim", 0),
            tensor_arg("index", "idx"),
            tensor_arg("src", "s"),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(op, TraceOp::ScatterAdd { dim: 0 }),
        "expected ScatterAdd {{ dim: 0 }}, got {op:?}"
    );
    assert_eq!(inputs, vec!["x", "idx", "s"]);
}

// -- Roll --

#[test]
fn test_map_roll() {
    let node = make_node(
        "torch.ops.aten.roll.default",
        vec![
            tensor_arg("self", "x"),
            ints_arg("shifts", &[2, -1]),
            ints_arg("dims", &[0, 1]),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    match &op {
        TraceOp::Roll { shifts, dims } => {
            assert_eq!(shifts, &[2, -1]);
            assert_eq!(dims, &[0, 1]);
        }
        _ => panic!("expected Roll, got {op:?}"),
    }
    assert_eq!(inputs, vec!["x"]);
}

// -- Conv3d --

#[test]
fn test_map_conv3d() {
    let ctx = ctx_with_weight(
        "w_conv3d",
        vec![0.01; 2 * 3 * 3 * 3 * 3],
        vec![2, 3, 3, 3, 3],
    );
    let node = make_node(
        "torch.ops.aten.conv3d.default",
        vec![
            tensor_arg("input", "x"),
            tensor_arg("weight", "w_conv3d"),
            ints_arg("stride", &[1, 1, 1]),
            ints_arg("padding", &[1, 1, 1]),
            ints_arg("dilation", &[1, 1, 1]),
            int_arg("groups", 1),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 5).unwrap();
    assert!(
        matches!(op, TraceOp::Conv3d { .. }),
        "expected Conv3d, got {op:?}"
    );
    if let TraceOp::Conv3d {
        padding,
        stride,
        dilation,
        groups,
        ..
    } = &op
    {
        assert_eq!(padding, &[1, 1, 1]);
        assert_eq!(stride, &[1, 1, 1]);
        assert_eq!(dilation, &[1, 1, 1]);
        assert_eq!(*groups, 1);
    }
    assert_eq!(inputs, vec!["x"]);
}

// -- Grid sample --

#[test]
fn test_map_grid_sample() {
    let node = make_node(
        "torch.ops.aten.grid_sample.default",
        vec![
            tensor_arg("self", "x"),
            tensor_arg("grid", "g"),
            int_arg("interpolation_mode", 0),
            int_arg("padding_mode", 0),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::GridSample {
                align_corners: false,
                ..
            }
        ),
        "expected GridSample, got {op:?}"
    );
    assert_eq!(inputs, vec!["x", "g"]);
}

#[test]
fn test_map_grid_sample_border_padding() {
    let node = make_node(
        "torch.ops.aten.grid_sample.default",
        vec![
            tensor_arg("self", "x"),
            tensor_arg("grid", "g"),
            int_arg("interpolation_mode", 0),
            int_arg("padding_mode", 1),
        ],
    );
    let ctx = empty_ctx();
    let (op, _) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    if let TraceOp::GridSample { padding_mode, .. } = &op {
        assert_eq!(
            *padding_mode,
            nn_core::dyn_tensor::GridSamplePaddingMode::Border
        );
    } else {
        panic!("expected GridSample, got {op:?}");
    }
}

#[test]
fn test_map_grid_sample_unsupported_nearest() {
    let node = make_node(
        "torch.ops.aten.grid_sample.default",
        vec![
            tensor_arg("self", "x"),
            tensor_arg("grid", "g"),
            int_arg("interpolation_mode", 1), // nearest, not supported
            int_arg("padding_mode", 0),
        ],
    );
    let ctx = empty_ctx();
    let result = map_node_to_trace_op(&node, &ctx, 4);
    assert!(
        result.is_err(),
        "expected error for unsupported nearest mode"
    );
}

// -- Masked fill (direct mapper, wave 11) --

#[test]
fn test_map_masked_fill_fallback() {
    // Wave 11 provides a direct mapper that encodes the fill value as a Custom op.
    // The try_expand_node path still decomposes into Constant + WhereCond when
    // shape metadata is available, but the single-op path no longer errors.
    let node = make_node(
        "torch.ops.aten.masked_fill.Scalar",
        vec![
            tensor_arg("self", "x"),
            tensor_arg("mask", "m"),
            float_arg("value", -1e9),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("masked_fill_scalar_")),
        "masked_fill should produce a custom op with scalar value, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x", "m"]);
}

// -- Index.Tensor (decomposition path) --

#[test]
fn test_map_index_tensor_fallback() {
    let node = make_node(
        "torch.ops.aten.index.Tensor",
        vec![tensor_arg("self", "x"), tensor_arg("indices", "idx")],
    );
    let ctx = empty_ctx();
    let result = map_node_to_trace_op(&node, &ctx, 0);
    assert!(
        result.is_err(),
        "index.Tensor should fail on single-op path (needs try_expand_node)"
    );
}

// -- Meshgrid (direct mapper, wave 11) --

#[test]
fn test_map_meshgrid_fallback() {
    // Wave 11 provides a direct mapper that produces a Custom op when
    // shape metadata is unavailable for the try_expand_node decomposition.
    let node = make_node(
        "torch.ops.aten.meshgrid.default",
        vec![tensor_arg("tensors", "x")],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 0).unwrap();
    assert!(
        matches!(&op, TraceOp::Custom { name } if name.starts_with("meshgrid_")),
        "meshgrid should produce a custom op, got: {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// -- Conv3d via convolution.default with 5D weights --

#[test]
fn test_map_convolution_5d_conv3d() {
    // aten.convolution.default with 5D weight should produce Conv3d.
    let ctx = ctx_with_weight("w5d", vec![0.01; 2 * 3 * 3 * 3], vec![2, 1, 3, 3, 3]);
    let node = make_node(
        "torch.ops.aten.convolution.default",
        vec![
            tensor_arg("input", "x"),
            tensor_arg("weight", "w5d"),
            ints_arg("stride", &[1, 1, 1]),
            ints_arg("padding", &[1, 1, 1]),
            ints_arg("dilation", &[1, 1, 1]),
            int_arg("groups", 1),
        ],
    );
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 5).unwrap();
    assert!(
        matches!(op, TraceOp::Conv3d { .. }),
        "expected Conv3d from convolution.default with 5D weights, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

// ---------------------------------------------------------------------------
// Supported ops coverage: verify all dpdf-needed aten ops are in supported_ops()
// ---------------------------------------------------------------------------

#[test]
fn test_dpdf_ops_in_supported_ops_list() {
    let supported = crate::supported_ops();

    // All ops needed by the 6 dpdf model architectures should be listed.
    let dpdf_ops = [
        // DocLayout-YOLO
        "aten::convolution",
        "aten::native_batch_norm",
        "aten::silu",
        "aten::relu",
        "aten::max_pool2d_with_indices",
        "aten::upsample_nearest2d",
        "aten::cat",
        "aten::hardswish",
        "aten::view",
        // Granite-Docling
        "aten::embedding",
        "aten::layer_norm",
        "aten::gelu",
        "aten::mm",
        "aten::softmax",
        "aten::scaled_dot_product_attention",
        "aten::linear",
        "aten::triu",
        // Table Transformer
        "aten::upsample_bilinear2d",
        "aten::tril",
        // Qwen3-VL
        "aten::rms_norm",
        // FireRed-OCR
        "aten::argmax",
        // Common vision
        "aten::transpose",
        "aten::permute",
        "aten::reshape",
        "aten::add",
        "aten::mul",
        "aten::sub",
        "aten::div",
        "aten::matmul",
        // Selection
        "aten::gather",
        "aten::split",
        "aten::split_with_sizes",
        "aten::unbind",
        // Misc
        "aten::mish",
        "aten::softplus",
        "aten::selu",
        "aten::hardsigmoid",
        "aten::pixel_shuffle",
        "aten::pixel_unshuffle",
        "aten::argmin",
        "aten::repeat",
        // Advanced indexing / shape
        "aten::stack",
        "aten::narrow",
        "aten::topk",
        "aten::sort",
        "aten::scatter_add",
        "aten::roll",
        // Vision model ops
        "aten::conv3d",
        "aten::grid_sample",
        "aten::meshgrid",
        "aten::index",
        "aten::masked_fill",
        // Vision model ops (conv_transpose2d, additional pooling)
        "aten::conv_transpose2d",
        "aten::max_pool2d",
        "aten::avg_pool1d",
        "aten::adaptive_avg_pool1d",
        "aten::adaptive_max_pool2d",
    ];

    for op in &dpdf_ops {
        assert!(
            supported.contains(op),
            "dpdf-required op '{op}' missing from supported_ops()"
        );
    }
}
