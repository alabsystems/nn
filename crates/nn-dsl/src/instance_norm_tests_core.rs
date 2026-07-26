// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Core InstanceNorm K2 tests: IR validation, shapes, pretty-print,
//! decomposed codegen, and reference implementation basic tests.

use super::*;
use crate::tensor_ir::tensor_ir_pretty_print;

#[test]
fn test_instance_norm_k2_validates() {
    let k2 = build_instance_norm(2, 4, 16).expect("build must succeed");
    k2.validate().expect("K2 InstanceNorm IR must validate");
}

#[test]
fn test_instance_norm_k2_zero_dim_returns_err() {
    let result = build_instance_norm(1, 0, 16);
    assert!(result.is_err(), "zero dimension must return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("must be positive"),
        "error message must mention 'must be positive', got: {err}"
    );
}

#[test]
fn test_instance_norm_k2_non_last_axis_rejected() {
    use crate::tensor_ir::{
        TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
    };
    // Manually construct a kernel with axis=0 (not the last axis) on [B, C, T].
    let kernel = TensorKernelDef::new(
        "instance_norm_bad_axis",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![2, 4, 16],
                },
                vec![2, 4, 16],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "eps".into(),
                    shape: vec![1],
                },
                vec![1],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::InstanceNorm1d {
                    input: TensorNodeId::new(0),
                    eps: TensorNodeId::new(1),
                    axis: 0, // NOT the last axis
                    gamma: None,
                    beta: None,
                },
                vec![2, 4, 16],
            ),
        ],
        TensorNodeId::new(2),
    );
    let err = kernel.validate().unwrap_err();
    assert!(
        matches!(
            err,
            TensorIRError::Layer(TensorIRLayerError::InstanceNormAxisNotLast { axis: 0, rank: 3 })
        ),
        "axis=0 on 3D tensor should be rejected, got: {err}"
    );
}

#[test]
fn test_instance_norm_k2_has_correct_output_shape() {
    let k2 = build_instance_norm(2, 4, 16).expect("build must succeed");
    let output_shape = &k2.nodes[k2.output.index()].shape;
    assert_eq!(output_shape, &[2, 4, 16]);
}

#[test]
fn test_instance_norm_k2_native_node_count() {
    let k2 = build_instance_norm(2, 4, 16).expect("build must succeed");
    assert_eq!(
        k2.nodes.len(),
        3,
        "native builder: 2 inputs + 1 InstanceNorm1d"
    );
}

#[test]
fn test_instance_norm_k2_native_pretty_print() {
    let k2 = build_instance_norm(1, 2, 4).expect("build must succeed");
    let ir = tensor_ir_pretty_print(&k2);
    assert!(ir.contains("tensor_kernel instance_norm"));
    assert!(ir.contains("instance_norm_1d(%0, eps=%1, axis=2)"));
    assert!(ir.contains("return %2"));
}

#[test]
fn test_instance_norm_k2_decomposed_node_count() {
    let k2 = build_instance_norm_decomposed(2, 4, 16).expect("build must succeed");
    assert_eq!(k2.nodes.len(), 12);
}

#[test]
fn test_instance_norm_k2_decomposed_pretty_print() {
    let k2 = build_instance_norm_decomposed(1, 2, 4).expect("build must succeed");
    let ir = tensor_ir_pretty_print(&k2);
    assert!(ir.contains("tensor_kernel instance_norm"));
    assert!(ir.contains("reduce_mean"));
    assert!(ir.contains("broadcast"));
    assert!(ir.contains("elementwise(rsqrt"));
    assert!(ir.contains("return %11"));
}

#[test]
fn test_instance_norm_k2_decomposed_msl_dispatch_plan() {
    use crate::codegen_msl_tensor::{build_dispatch_plan, DispatchStep};

    let k2 = build_instance_norm_decomposed(2, 4, 16).expect("build must succeed");
    let (plan, _) = build_dispatch_plan(&k2, ScalarType::F32).expect("dispatch plan must succeed");

    let reduce_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Reduce { .. }))
        .count();
    let ew_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Elementwise { .. }))
        .count();
    let bc_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Broadcast { .. }))
        .count();

    assert_eq!(reduce_count, 2, "2 reductions (mean(x), mean((x-mean)²))");
    assert_eq!(
        ew_count, 5,
        "5 element-wise ops (sub, square, add, rsqrt, mul)"
    );
    assert_eq!(bc_count, 3, "3 broadcasts (mean, var, eps)");
}

#[test]
fn test_instance_norm_k2_decomposed_msl_codegen() {
    let k2 = build_instance_norm_decomposed(2, 4, 16).expect("build must succeed");
    let msl = crate::codegen_msl_tensor_emit::emit_tensor_msl(&k2, ScalarType::F32)
        .expect("MSL codegen must succeed");

    assert!(msl.contains("reduce_dim"));
    assert!(msl.contains("threadgroup_barrier"));
    assert!(msl.contains("threadgroup float shared[256]"));
}

#[test]
fn test_instance_norm_ref_constant_input() {
    let x = vec![5.0f32; 2 * 3 * 8];
    let out = instance_norm_ref(&x, 2, 3, 8, 1e-5).expect("ref must succeed");
    for &v in &out {
        assert!(
            v.abs() < 1e-3,
            "constant input should normalize to ~0, got {v}"
        );
    }
}

#[test]
fn test_instance_norm_ref_known_values() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let eps = 1e-5;
    let out = instance_norm_ref(&x, 1, 1, 4, eps).expect("ref must succeed");

    let mean = 2.5;
    let var = 1.25;
    let inv_std = 1.0 / (var + eps).sqrt();
    let expected: Vec<f32> = x.iter().map(|v| (v - mean) * inv_std).collect();

    for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "mismatch at index {i}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_instance_norm_ref_output_has_zero_mean() {
    let x: Vec<f32> = (0..24).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let out = instance_norm_ref(&x, 1, 3, 8, 1e-5).expect("ref must succeed");

    for ci in 0..3 {
        let channel: &[f32] = &out[ci * 8..(ci + 1) * 8];
        let mean: f32 = channel.iter().sum::<f32>() / 8.0;
        assert!(
            mean.abs() < 1e-5,
            "channel {ci} mean should be ~0, got {mean}"
        );
    }
}

#[test]
fn test_instance_norm_ref_output_has_unit_variance() {
    let x: Vec<f32> = (0..24).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let out = instance_norm_ref(&x, 1, 3, 8, 1e-5).expect("ref must succeed");

    for ci in 0..3 {
        let channel: &[f32] = &out[ci * 8..(ci + 1) * 8];
        let mean: f32 = channel.iter().sum::<f32>() / 8.0;
        let var: f32 = channel.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 8.0;
        assert!(
            (var - 1.0).abs() < 0.01,
            "channel {ci} variance should be ~1, got {var}"
        );
    }
}

#[test]
fn test_instance_norm_ref_multi_batch() {
    let mut x = vec![0.0f32; 8]; // 2 batches × 1 channel × 4 time
    x[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    x[4..8].copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);

    let out = instance_norm_ref(&x, 2, 1, 4, 1e-5).expect("ref must succeed");

    // Same relative spacing → same normalized output for both batches
    for i in 0..4 {
        assert!(
            (out[i] - out[4 + i]).abs() < 1e-5,
            "batch0[{i}]={} != batch1[{i}]={}",
            out[i],
            out[4 + i]
        );
    }
}

#[test]
fn test_instance_norm_ref_zero_dim_returns_err() {
    let result = instance_norm_ref(&[], 0, 1, 1, 1e-5);
    assert!(result.is_err(), "zero dimension must return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("must be positive"),
        "error message must mention 'must be positive', got: {err}"
    );
}

#[test]
fn test_instance_norm_ref_wrong_length_returns_err() {
    let result = instance_norm_ref(&[1.0, 2.0, 3.0], 1, 1, 4, 1e-5);
    assert!(result.is_err(), "wrong length must return Err");
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("shape mismatch"),
        "error message must mention 'shape mismatch', got: {err}"
    );
}
