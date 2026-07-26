// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `DispatchBuilder`.

use super::*;
use nn_dsl::DispatchStep;

#[test]
fn test_builder_linear_node_count() {
    let mut b = DispatchBuilder::new();
    b.linear("test_linear", 64, 128, 1);
    assert_eq!(b.node_count(), 4, "Linear allocates 4 nodes");
    let steps = b.into_steps();
    assert_eq!(steps.len(), 1);
    assert!(
        matches!(&steps[0], DispatchStep::Linear { kernel_name, .. } if kernel_name == "test_linear")
    );
}

#[test]
fn test_builder_sigmoid_node_count() {
    let mut b = DispatchBuilder::new();
    b.sigmoid("test_sig", 256);
    assert_eq!(b.node_count(), 2, "Sigmoid allocates 2 nodes");
    let steps = b.into_steps();
    assert_eq!(steps.len(), 1);
    assert!(matches!(
        &steps[0],
        DispatchStep::Sigmoid {
            total_elements: 256,
            ..
        }
    ));
}

#[test]
fn test_builder_binary_ops_node_count() {
    let mut b = DispatchBuilder::new();
    b.binary_add("add", 128);
    b.binary_mul("mul", 64);
    assert_eq!(b.node_count(), 6, "Two binary ops = 2*3 = 6 nodes");
    assert_eq!(b.into_steps().len(), 2);
}

#[test]
fn test_builder_conv1d_fields() {
    let mut b = DispatchBuilder::new();
    b.conv1d("conv_pre", 512, 512, 7, 100, 1, 3, 1);
    assert_eq!(b.node_count(), 4, "Conv1d allocates 4 nodes");
    let steps = b.into_steps();
    assert_eq!(steps.len(), 1);
    assert!(matches!(&steps[0], DispatchStep::Conv1d(..)));
}

#[test]
fn test_builder_matmul_total_elements() {
    let mut b = DispatchBuilder::new();
    b.matmul("test_mm", 8, 64, 32, 2, false, false, None);
    let steps = b.into_steps();
    assert!(matches!(
        &steps[0],
        DispatchStep::MatMul { total_elements, .. } if *total_elements == 2 * 8 * 32
    ));
}

#[test]
fn test_builder_embedding_node_count() {
    let mut b = DispatchBuilder::new();
    b.embedding("emb", 768, 10);
    assert_eq!(b.node_count(), 3, "Embedding allocates 3 nodes");
    let steps = b.into_steps();
    assert!(matches!(
        &steps[0],
        DispatchStep::Embedding { total_elements, .. } if *total_elements == 10 * 768
    ));
}

#[test]
fn test_builder_with_capacity() {
    let b = DispatchBuilder::with_capacity(256);
    assert_eq!(b.node_count(), 0);
    assert_eq!(b.into_steps().len(), 0);
}

#[test]
fn test_builder_mixed_sequence() {
    let mut b = DispatchBuilder::with_capacity(16);
    b.embedding("emb", 768, 10); // 3 nodes
    b.linear("fc1", 768, 256, 10); // 4 nodes
    b.gelu("act", 2560); // 2 nodes
    b.linear("fc2", 256, 128, 10); // 4 nodes
    b.softmax("sm", 128, 10); // 2 nodes
    assert_eq!(b.node_count(), 15);
    assert_eq!(b.into_steps().len(), 5);
}
