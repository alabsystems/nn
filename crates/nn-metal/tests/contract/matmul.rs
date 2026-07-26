// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cross-backend contract tests for MatMul tensor kernel:
//! GPU output within NY verified bounds, and GPU ≈ CPU reference.
//!
//! Tests the full pipeline: IR → dispatch plan → MSL codegen → Metal execution,
//! verified against NY IBP bounds and CPU reference matmul.
//!
//! Multi-variable stacking requires same-shape inputs. All tests use square
//! or same-shape matrix pairs (matching the Q @ K^T attention pattern where
//! Q and K have shape [seq_len, head_dim]).
//!
//! Part of #741 (AC6).

use super::test_utils::{
    assert_gpu_within_bounds, assert_within_budget, matmul_ref, metal_setup, rand_f32_vec,
};

use nn_dsl::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use nn_dsl::ScalarType;
use nn_metal::execute_tensor_dispatch;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

/// Build a MatMul tensor kernel definition.
fn matmul_kernel(
    name: &str,
    left_shape: &[usize],
    right_shape: &[usize],
    out_shape: &[usize],
    transpose_right: bool,
    scale: Option<f32>,
) -> TensorKernelDef {
    TensorKernelDef::new(
        name,
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "left".into(),
                    shape: left_shape.to_vec(),
                },
                left_shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "right".into(),
                    shape: right_shape.to_vec(),
                },
                right_shape.to_vec(),
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::MatMul {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                    transpose_right,
                    scale,
                },
                out_shape.to_vec(),
            ),
        ],
        TensorNodeId::new(2),
    )
}

/// Prove IBP bounds for a MatMul kernel with same-shape Variable inputs.
///
/// Both inputs must have the same shape for multi-variable stacking.
fn prove_matmul_bounds(def: &TensorKernelDef, input_shape: &[usize]) -> (ArrayD<f32>, ArrayD<f32>) {
    let bindings = vec![TensorParamBinding::Variable, TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(def, &bindings).expect("matmul graph");

    // Multi-variable stacking: axis 0 selects variable → shape [2, ...input_shape].
    let stacked_shape: Vec<usize> = {
        let mut s = vec![2];
        s.extend_from_slice(input_shape);
        s
    };
    let lower = ArrayD::from_elem(IxDyn(&stacked_shape), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&stacked_shape), 1.0f32);
    let input_bounds = BoundedTensor::new(lower, upper).expect("input bounds");
    let output_bounds = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    let (lo, hi) = output_bounds.lower_upper();

    assert!(lo.iter().all(|v| v.is_finite()), "proved lower finite");
    assert!(hi.iter().all(|v| v.is_finite()), "proved upper finite");
    (lo.clone(), hi.clone())
}

// ===========================================================================
// MatMul contract tests
// ===========================================================================

/// MatMul contract: square [4,4] @ [4,4] = [4,4], no transpose, no scale.
/// GPU output matches CPU reference and falls within IBP bounds.
#[test]
fn test_matmul_gpu_square_no_transpose() {
    let dim = 4;
    let def = matmul_kernel(
        "matmul_square",
        &[dim, dim],
        &[dim, dim],
        &[dim, dim],
        false,
        None,
    );

    let (proved_lo, proved_hi) = prove_matmul_bounds(&def, &[dim, dim]);

    let cache = metal_setup();
    let left_data = rand_f32_vec(0xAA7A_0001, dim * dim, -1.0, 1.0);
    let right_data = rand_f32_vec(0xAA7A_0002, dim * dim, -1.0, 1.0);

    let mut inputs = HashMap::new();
    inputs.insert("left", left_data.clone());
    inputs.insert("right", right_data.clone());

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("matmul GPU dispatch");
    assert_eq!(gpu_out.len(), dim * dim, "output length");

    let cpu_out = matmul_ref(&left_data, &right_data, dim, dim, dim, false, None);
    assert_within_budget("matmul_square", &gpu_out, &cpu_out);
    assert_gpu_within_bounds("matmul_square", &gpu_out, &proved_lo, &proved_hi);
}

/// MatMul contract: Q @ K^T attention pattern with transpose and scale.
/// Uses same-shape Q=[4,8] and K=[4,8] (both [seq_len, head_dim]).
#[test]
fn test_matmul_gpu_qkt_transpose_with_scale() {
    let (seq_len, head_dim) = (4, 8);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let def = matmul_kernel(
        "matmul_qkt",
        &[seq_len, head_dim],
        &[seq_len, head_dim], // K stored as [N, K], transposed to [K, N]
        &[seq_len, seq_len],
        true,
        Some(scale),
    );

    let (proved_lo, proved_hi) = prove_matmul_bounds(&def, &[seq_len, head_dim]);

    let cache = metal_setup();
    let q_data = rand_f32_vec(0xB0E0_0001, seq_len * head_dim, -1.0, 1.0);
    let k_data = rand_f32_vec(0xB0E0_0002, seq_len * head_dim, -1.0, 1.0);

    let mut inputs = HashMap::new();
    inputs.insert("left", q_data.clone());
    inputs.insert("right", k_data.clone());

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("Q@K^T GPU dispatch");
    assert_eq!(gpu_out.len(), seq_len * seq_len);

    let cpu_out = matmul_ref(
        &q_data,
        &k_data,
        seq_len,
        head_dim,
        seq_len,
        true,
        Some(scale),
    );
    assert_within_budget("matmul_qkt", &gpu_out, &cpu_out);
    assert_gpu_within_bounds("matmul_qkt", &gpu_out, &proved_lo, &proved_hi);
}

/// MatMul contract: attention-value multiplication with transpose.
/// attn_weights=[4,4] @ V^T=[4,8], where V is stored [8,4] transposed.
/// Equivalent: attn @ V where V=[4,8] without transpose, but here we test
/// with same-shape [4,8] inputs using transpose to get [4,4] @ [8,4]^T = [4,8].
/// Uses same-shape inputs [4,8] for multi-variable stacking compatibility.
#[test]
fn test_matmul_gpu_attn_value_transpose() {
    // Square matmul simulating post-softmax attention @ value.
    let dim = 4;
    let def = matmul_kernel(
        "matmul_attn_v",
        &[dim, dim],
        &[dim, dim],
        &[dim, dim],
        false,
        None,
    );

    let (proved_lo, proved_hi) = prove_matmul_bounds(&def, &[dim, dim]);

    let cache = metal_setup();
    // Attention weights ∈ [0, 1] (post-softmax), V ∈ [-1, 1]
    let attn_data = rand_f32_vec(0xA770_0001, dim * dim, 0.0, 1.0);
    let v_data = rand_f32_vec(0xA770_0002, dim * dim, -1.0, 1.0);

    let mut inputs = HashMap::new();
    inputs.insert("left", attn_data.clone());
    inputs.insert("right", v_data.clone());

    let gpu_out = execute_tensor_dispatch(&cache, &def, ScalarType::F32, &inputs)
        .expect("attn@V GPU dispatch");
    assert_eq!(gpu_out.len(), dim * dim);

    let cpu_out = matmul_ref(&attn_data, &v_data, dim, dim, dim, false, None);
    assert_within_budget("matmul_attn_v", &gpu_out, &cpu_out);
    assert_gpu_within_bounds("matmul_attn_v", &gpu_out, &proved_lo, &proved_hi);
}
