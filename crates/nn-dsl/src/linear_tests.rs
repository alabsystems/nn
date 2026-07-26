// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `linear.rs` — Linear layer builder.

use super::*;
use crate::tensor_ir::{TensorIRError, TensorIRLayerError, TensorOpKind};

#[test]
fn test_build_linear_no_bias() {
    let def = build_linear("fc1", 768, 3072, false).expect("valid linear");
    assert_eq!(def.name, "fc1");
    assert_eq!(def.nodes.len(), 3); // data, weight, linear
    assert_eq!(def.nodes[0].shape, vec![768]);
    assert_eq!(def.nodes[1].shape, vec![3072, 768]);
    assert_eq!(def.nodes[2].shape, vec![3072]);
    assert!(matches!(
        &def.nodes[2].kind,
        TensorOpKind::Linear { bias: None, .. }
    ));
}

#[test]
fn test_build_linear_with_bias() {
    let def = build_linear("fc2", 128, 512, true).expect("valid linear with bias");
    assert_eq!(def.nodes.len(), 4); // data, weight, bias, linear
    assert_eq!(def.nodes[2].shape, vec![512]); // bias shape
    assert_eq!(def.nodes[3].shape, vec![512]); // output shape
    assert!(matches!(
        &def.nodes[3].kind,
        TensorOpKind::Linear { bias: Some(_), .. }
    ));
}

#[test]
fn test_build_linear_batched_no_bias() {
    let def = build_linear_batched("attn_proj", 8, 768, 768, false).expect("valid batched linear");
    assert_eq!(def.nodes.len(), 3); // data, weight, linear
    assert_eq!(def.nodes[0].shape, vec![8, 768]); // [batch, in_features]
    assert_eq!(def.nodes[1].shape, vec![768, 768]); // [out, in]
    assert_eq!(def.nodes[2].shape, vec![8, 768]); // [batch, out_features]
}

#[test]
fn test_build_linear_batched_with_bias() {
    let def = build_linear_batched("mlp", 16, 1024, 4096, true).expect("valid batched with bias");
    assert_eq!(def.nodes.len(), 4);
    assert_eq!(def.nodes[0].shape, vec![16, 1024]);
    assert_eq!(def.nodes[1].shape, vec![4096, 1024]);
    assert_eq!(def.nodes[2].shape, vec![4096]); // bias
    assert_eq!(def.nodes[3].shape, vec![16, 4096]); // output
}

#[test]
fn test_build_linear_zero_in_features_rejected() {
    let err = build_linear("bad", 0, 512, false).unwrap_err();
    assert!(matches!(
        err,
        TensorIRError::Layer(TensorIRLayerError::LinearInputScalar)
    ));
}

#[test]
fn test_build_linear_zero_out_features_rejected() {
    let err = build_linear("bad", 768, 0, false).unwrap_err();
    assert!(matches!(
        err,
        TensorIRError::Layer(TensorIRLayerError::LinearWeightNotMatrix { .. })
    ));
}

#[test]
fn test_build_linear_batched_zero_batch_rejected() {
    let err = build_linear_batched("bad", 0, 768, 512, false).unwrap_err();
    assert!(matches!(err, TensorIRError::EmptyDimension(_)));
}

#[test]
fn test_build_linear_dvoice_representative_dims() {
    // Qwen3-TTS dimensions: 2048 -> 6144 (MLP up-projection)
    let def =
        build_linear_batched("qwen_mlp_up", 1, 2048, 6144, false).expect("dvoice-scale linear");
    assert_eq!(def.nodes[0].shape, vec![1, 2048]);
    assert_eq!(def.nodes[1].shape, vec![6144, 2048]);
    assert_eq!(def.nodes[2].shape, vec![1, 6144]);
}

#[test]
fn test_build_linear_pretty_print() {
    let def = build_linear("fc", 64, 128, true).expect("valid");
    let pretty = crate::tensor_ir::tensor_ir_pretty_print(&def);
    assert!(
        pretty.contains("linear("),
        "should contain linear op: {pretty}"
    );
    assert!(
        pretty.contains("weight="),
        "should show weight ref: {pretty}"
    );
    assert!(pretty.contains("bias="), "should show bias ref: {pretty}");
}

#[test]
fn test_build_linear_output_node_is_last() {
    let def = build_linear("fc", 256, 512, true).expect("valid");
    assert_eq!(def.output.index(), def.nodes.len() - 1);
}
