// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for peephole pass 8: AddLayerNorm + Linear → AddNormLinear.

use super::super::super::{CompiledKernel, CompiledStep, NativeOpKind};
use std::collections::HashMap;

use nn_core::dyn_tensor::trace::WeightRef;

/// Helper: make a minimal AddLayerNorm NativeOp.
fn make_add_layer_norm(hidden_dim: usize) -> CompiledStep {
    let mut wd = HashMap::new();
    wd.insert("weight".to_string(), WeightRef::from_shape(&[hidden_dim]));
    wd.insert("bias".to_string(), WeightRef::from_shape(&[hidden_dim]));
    CompiledStep::NativeOp {
        op: NativeOpKind::AddLayerNorm {
            eps: 1e-5,
            input_shape: vec![2, 16, hidden_dim],
            hidden_dim,
        },
        weight_data: wd,
    }
}

/// Helper: build a Linear Dispatch step with the correct IR.
fn make_linear_dispatch(
    input_shape: &[usize],
    out_features: usize,
    has_bias: bool,
) -> CompiledStep {
    use crate::tensor_block_builder::TensorBlockBuilder;

    let in_features = *input_shape.last().unwrap();
    let mut output_shape = input_shape.to_vec();
    *output_shape.last_mut().unwrap() = out_features;
    let weight_shape = [out_features, in_features];

    let mut b = TensorBlockBuilder::new("linear");
    let input = b.add_input("input_0", input_shape);
    let w = b.add_input("weight", &weight_shape);
    let bi = if has_bias {
        Some(b.add_input("bias", &[out_features]))
    } else {
        None
    };
    let output = b.add_linear(input, w, bi, &output_shape);
    let def = b.build(output).expect("valid linear IR");

    let mut weight_data = HashMap::new();
    weight_data.insert(
        "weight".to_string(),
        WeightRef::new(
            vec![0.0f32; out_features * in_features],
            weight_shape.to_vec(),
        )
        .expect("valid weight"),
    );
    if has_bias {
        weight_data.insert(
            "bias".to_string(),
            WeightRef::new(vec![0.0f32; out_features], vec![out_features]).expect("valid bias"),
        );
    }

    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: None,
    }
}

#[test]
fn test_add_norm_linear_variant_name_and_dispatch_count() {
    let op = NativeOpKind::AddNormLinear {
        eps: 1e-5,
        input_shape: vec![2, 16, 768],
        hidden_dim: 768,
        out_features: 768,
        has_bias: true,
    };
    assert_eq!(op.variant_name(), "AddNormLinear");
    // Simdgroup eligible: m=32, k=768, n=768 → 2 dispatches.
    assert_eq!(op.estimated_metal_dispatches(), 2);
}

#[test]
fn test_fuse_add_norm_linear_basic() {
    // AddLayerNorm(hidden=768) + Linear(768→3072) → AddNormLinear.
    let hidden_dim = 768;
    let out_features = 3072;
    let input_shape = [2, 16, hidden_dim];

    let mut steps = vec![
        make_add_layer_norm(hidden_dim),
        make_linear_dispatch(&input_shape, out_features, true),
    ];
    // use_counts: step 0 consumed by step 1 (fan-out = 1).
    let use_counts = vec![1, 0];

    super::fuse_add_norm_linear(&mut steps, &use_counts);

    // Step 0 should be AddNormLinear.
    match &steps[0] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::AddNormLinear {
                    eps,
                    hidden_dim: hd,
                    out_features: of,
                    has_bias,
                    input_shape: is,
                },
            weight_data,
        } => {
            assert!((eps - 1e-5).abs() < 1e-10);
            assert_eq!(*hd, hidden_dim);
            assert_eq!(*of, out_features);
            assert!(*has_bias);
            assert_eq!(is, &input_shape);
            // Norm weights renamed.
            assert!(
                weight_data.contains_key("norm_weight"),
                "norm_weight missing"
            );
            assert!(weight_data.contains_key("norm_bias"), "norm_bias missing");
            // Linear weights preserved.
            assert!(weight_data.contains_key("weight"), "linear weight missing");
            assert!(weight_data.contains_key("bias"), "linear bias missing");
        }
        other => panic!(
            "expected AddNormLinear at step 0, got: {:?}",
            std::mem::discriminant(other)
        ),
    }

    // Step 1 should be IdentityPassthrough (linear consumed).
    assert!(
        matches!(&steps[1], CompiledStep::IdentityPassthrough),
        "expected IdentityPassthrough at step 1"
    );
}

#[test]
fn test_fuse_add_norm_linear_no_bias() {
    // AddLayerNorm(hidden=256) + Linear(256→512, no bias) → AddNormLinear.
    let hidden_dim = 256;
    let out_features = 512;
    let input_shape = [2, 16, hidden_dim];

    let mut steps = vec![
        make_add_layer_norm(hidden_dim),
        make_linear_dispatch(&input_shape, out_features, false),
    ];
    let use_counts = vec![1, 0];

    super::fuse_add_norm_linear(&mut steps, &use_counts);

    match &steps[0] {
        CompiledStep::NativeOp {
            op:
                NativeOpKind::AddNormLinear {
                    has_bias,
                    out_features: of,
                    ..
                },
            weight_data,
        } => {
            assert!(!has_bias);
            assert_eq!(*of, out_features);
            assert!(!weight_data.contains_key("bias"), "no bias expected");
        }
        other => panic!(
            "expected AddNormLinear at step 0, got: {:?}",
            std::mem::discriminant(other)
        ),
    }
    assert!(matches!(&steps[1], CompiledStep::IdentityPassthrough));
}

#[test]
fn test_fuse_add_norm_linear_rejected_fanout_gt_1() {
    // AddLayerNorm with fan-out > 1 must NOT fuse.
    let hidden_dim = 768;
    let input_shape = [2, 16, hidden_dim];

    let mut steps = vec![
        make_add_layer_norm(hidden_dim),
        make_linear_dispatch(&input_shape, 3072, true),
    ];
    // Fan-out = 2 for step 0 → should not fuse.
    let use_counts = vec![2, 0];

    super::fuse_add_norm_linear(&mut steps, &use_counts);

    // Step 0 should remain AddLayerNorm.
    assert!(
        matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::AddLayerNorm { .. },
                ..
            }
        ),
        "AddLayerNorm should remain unfused when fan-out > 1"
    );
    // Step 1 should remain Dispatch (not IdentityPassthrough).
    assert!(
        matches!(&steps[1], CompiledStep::Dispatch { .. }),
        "Linear should remain as Dispatch when fusion blocked"
    );
}

#[test]
fn test_fuse_add_norm_linear_rejected_hidden_dim_too_large() {
    // hidden_dim > 7680 exceeds threadgroup memory limit → must NOT fuse.
    let hidden_dim = 8192;
    let input_shape = [2, 16, hidden_dim];

    let mut steps = vec![
        make_add_layer_norm(hidden_dim),
        make_linear_dispatch(&input_shape, 8192, true),
    ];
    let use_counts = vec![1, 0];

    super::fuse_add_norm_linear(&mut steps, &use_counts);

    // Step 0 should remain AddLayerNorm — hidden_dim > 7680.
    assert!(
        matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::AddLayerNorm { .. },
                ..
            }
        ),
        "AddLayerNorm should remain unfused when hidden_dim > 7680"
    );
    assert!(
        matches!(&steps[1], CompiledStep::Dispatch { .. }),
        "Linear should remain as Dispatch when fusion blocked by hidden_dim"
    );
}

#[test]
fn test_fuse_add_norm_linear_rejected_dim_mismatch() {
    // AddLayerNorm(hidden=768) + Linear(in_features=512) → dim mismatch, no fuse.
    let hidden_dim = 768;
    // Deliberately use a different in_features for the linear.
    let linear_input_shape = [2, 16, 512];

    let mut steps = vec![
        make_add_layer_norm(hidden_dim),
        make_linear_dispatch(&linear_input_shape, 1024, true),
    ];
    let use_counts = vec![1, 0];

    super::fuse_add_norm_linear(&mut steps, &use_counts);

    // Should remain unfused: hidden_dim 768 != in_features 512.
    assert!(
        matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::AddLayerNorm { .. },
                ..
            }
        ),
        "should not fuse when hidden_dim != linear in_features"
    );
}

#[test]
fn test_fuse_add_norm_linear_boundary_hidden_dim_7680() {
    // hidden_dim = 7680 is exactly at the limit → should fuse.
    let hidden_dim = 7680;
    let input_shape = [1, 4, hidden_dim];

    let mut steps = vec![
        make_add_layer_norm(hidden_dim),
        make_linear_dispatch(&input_shape, 1024, true),
    ];
    let use_counts = vec![1, 0];

    super::fuse_add_norm_linear(&mut steps, &use_counts);

    assert!(
        matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::AddNormLinear {
                    hidden_dim: 7680,
                    ..
                },
                ..
            }
        ),
        "hidden_dim=7680 should fuse (boundary)"
    );
    assert!(matches!(&steps[1], CompiledStep::IdentityPassthrough));
}

#[test]
fn test_fuse_add_norm_linear_boundary_hidden_dim_7681() {
    // hidden_dim = 7681 exceeds the limit → should NOT fuse.
    let hidden_dim = 7681;
    let input_shape = [1, 4, hidden_dim];

    let mut steps = vec![
        make_add_layer_norm(hidden_dim),
        make_linear_dispatch(&input_shape, 1024, true),
    ];
    let use_counts = vec![1, 0];

    super::fuse_add_norm_linear(&mut steps, &use_counts);

    assert!(
        matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::AddLayerNorm { .. },
                ..
            }
        ),
        "hidden_dim=7681 should NOT fuse (exceeds 7680 limit)"
    );
}

#[test]
fn test_fuse_add_norm_linear_skips_non_linear_dispatch() {
    // AddLayerNorm followed by non-linear Dispatch (e.g., "relu") → no fuse.
    let hidden_dim = 768;

    use crate::tensor_block_builder::TensorBlockBuilder;

    let shape = [2, 16, hidden_dim];
    let mut b = TensorBlockBuilder::new("relu");
    let input = b.add_input("input_0", &shape);
    let output = b.add_relu(input, &shape);
    let def = b.build(output).expect("valid relu IR");

    let relu_step = CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    };

    let mut steps = vec![make_add_layer_norm(hidden_dim), relu_step];
    let use_counts = vec![1, 0];

    super::fuse_add_norm_linear(&mut steps, &use_counts);

    // Should remain unfused: step[1] is "relu", not "linear".
    assert!(
        matches!(
            &steps[0],
            CompiledStep::NativeOp {
                op: NativeOpKind::AddLayerNorm { .. },
                ..
            }
        ),
        "should not fuse AddLayerNorm + non-linear dispatch"
    );
}

#[test]
fn test_fuse_add_norm_linear_empty_steps() {
    // Edge case: empty or too-short step arrays don't panic.
    let mut empty: Vec<CompiledStep> = vec![];
    super::fuse_add_norm_linear(&mut empty, &[]);

    let mut single = vec![make_add_layer_norm(768)];
    super::fuse_add_norm_linear(&mut single, &[0]);
    // No fusion possible, no panic.
    assert!(matches!(
        &single[0],
        CompiledStep::NativeOp {
            op: NativeOpKind::AddLayerNorm { .. },
            ..
        }
    ));
}
