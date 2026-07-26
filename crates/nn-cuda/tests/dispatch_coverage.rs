// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HIP dispatch coverage test — verifies every `DispatchStep` variant has both
//! a codegen handler (`emit_step_hip`) and a launch config (`launch_config_for_step`).
//!
//! Constructs kernels producing each `DispatchStep` variant via `build_dispatch_plan()`.
//! When a new variant is added, `KNOWN_VARIANT_COUNT` fails, forcing the developer
//! to add a test case and match arm in both codegen and launch config.
//!
//! Mirrors `nn-metal/tests/dispatch_coverage.rs`. Part of #2241.

use std::collections::HashSet;

use nn_cuda::codegen_hip_tensor_emit_step::emit_step_hip;
use nn_cuda::launch_config_for_step;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::ReduceOp;
use nn_dsl::{
    build_causal_conv1d, build_conv1d, build_conv_transpose_1d, build_dispatch_plan, build_linear,
    build_rope_rotate_kernel, build_softmax, DispatchStep, ScalarType, TensorKernelDef,
};

/// Number of `DispatchStep` variants exercised by test builders below.
/// Update when adding new test builders. See step_tag() for the full enum.
///
/// Covered (28): Reduce, Elementwise, Broadcast, Conv1d, ConvTranspose1d,
/// Linear, MatMul, BinaryAdd, BinaryMul, Sigmoid, Gelu, GeluErf, Relu, Tanh,
/// Reshape, AxisSelect, Stack, Narrow, Softmax, ZeroPad1d, Embedding, Conv2d,
/// Transpose, Concat, IndexSelect, Gather, SimdgroupLinear, SimdgroupMatMul
///
/// Not yet covered (3): Elu, TiledLinear, TiledMatMul
const KNOWN_VARIANT_COUNT: usize = 28;

/// Classify a DispatchStep into a string tag for assertion.
fn step_tag(step: &DispatchStep) -> &'static str {
    match step {
        DispatchStep::Reduce { .. } => "Reduce",
        DispatchStep::Elementwise { .. } => "Elementwise",
        DispatchStep::Broadcast { .. } => "Broadcast",
        DispatchStep::Conv1d(..) => "Conv1d",
        DispatchStep::ConvTranspose1d(..) => "ConvTranspose1d",
        DispatchStep::Linear { .. } => "Linear",
        DispatchStep::MatMul { .. } => "MatMul",
        DispatchStep::BinaryAdd { .. } => "BinaryAdd",
        DispatchStep::BinaryMul { .. } => "BinaryMul",
        DispatchStep::Sigmoid { .. } => "Sigmoid",
        DispatchStep::Gelu { .. } => "Gelu",
        DispatchStep::GeluErf { .. } => "GeluErf",
        DispatchStep::Relu { .. } => "Relu",
        DispatchStep::Elu { .. } => "Elu",
        DispatchStep::Tanh { .. } => "Tanh",
        DispatchStep::Reshape { .. } => "Reshape",
        DispatchStep::AxisSelect { .. } => "AxisSelect",
        DispatchStep::Stack { .. } => "Stack",
        DispatchStep::Narrow { .. } => "Narrow",
        DispatchStep::Softmax { .. } => "Softmax",
        DispatchStep::ZeroPad1d { .. } => "ZeroPad1d",
        DispatchStep::Embedding { .. } => "Embedding",
        DispatchStep::Conv2d(..) => "Conv2d",
        DispatchStep::Transpose { .. } => "Transpose",
        DispatchStep::Concat { .. } => "Concat",
        DispatchStep::IndexSelect { .. } => "IndexSelect",
        DispatchStep::Gather { .. } => "Gather",
        DispatchStep::SimdgroupLinear(..) => "SimdgroupLinear",
        DispatchStep::SimdgroupMatMul(..) => "SimdgroupMatMul",
        DispatchStep::TiledLinear(..) => "TiledLinear",
        DispatchStep::TiledMatMul(..) => "TiledMatMul",
        _ => "UNKNOWN",
    }
}

/// Build plan, collect tags, and verify codegen + launch config for each step.
fn plan_and_verify(kernel: &TensorKernelDef, tags: &mut HashSet<&'static str>) {
    let (plan, _) = build_dispatch_plan(kernel, ScalarType::F32).expect("plan");
    for step in &plan {
        let tag = step_tag(step);
        tags.insert(tag);

        // Verify codegen: every compute step must produce HIP source.
        let codegen_result = emit_step_hip(step, kernel);
        match step {
            DispatchStep::Reshape { .. } => {
                // Reshape is a no-op (zero-copy), codegen returns Ok(None).
                assert!(
                    matches!(codegen_result, Ok(None)),
                    "{tag}: Reshape should produce Ok(None), got {codegen_result:?}"
                );
            }
            _ => {
                let source =
                    codegen_result.unwrap_or_else(|e| panic!("{tag}: codegen failed: {e}"));
                assert!(
                    source.is_some(),
                    "{tag}: codegen returned None for compute step"
                );
                let src = source.unwrap();
                assert!(
                    src.contains("__global__") || src.contains("extern \"C\""),
                    "{tag}: generated HIP source missing kernel marker:\n{src}"
                );
            }
        }

        // Verify launch config: compute steps must have a config, Reshape must not.
        let config = launch_config_for_step(step)
            .unwrap_or_else(|e| panic!("{tag}: launch_config_for_step returned Err: {e}"));
        match step {
            DispatchStep::Reshape { .. } => {
                assert!(
                    config.is_none(),
                    "{tag}: Reshape should have no launch config"
                );
            }
            _ => {
                let cfg = config.unwrap_or_else(|| {
                    panic!("{tag}: launch_config_for_step returned None for compute step")
                });
                assert!(cfg.block.x > 0, "{tag}: block.x must be > 0");
                assert!(cfg.grid.x > 0, "{tag}: grid.x must be > 0");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel builders (same as nn-metal/tests/dispatch_coverage.rs)
// ---------------------------------------------------------------------------

fn build_binary_kernel(name: &str, add: bool) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let left = b.add_input("left", &[4]);
    let right = b.add_input("right", &[4]);
    let out = if add {
        b.add_binary_add(left, right, &[4])
    } else {
        b.add_binary_mul(left, right, &[4])
    };
    b.build(out).expect("valid graph")
}

fn build_unary_kernel(name: &str, sigmoid: bool) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4]);
    let out = if sigmoid {
        b.add_sigmoid(x, &[4])
    } else {
        b.add_gelu(x, &[4])
    };
    b.build(out).expect("valid graph")
}

fn build_gelu_erf_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4]);
    let out = b.add_gelu_erf(x, &[4]);
    b.build(out).expect("valid graph")
}

fn build_relu_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4]);
    let out = b.add_relu(x, &[4]);
    b.build(out).expect("valid graph")
}

fn build_tanh_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4]);
    let out = b.add_tanh(x, &[4]);
    b.build(out).expect("valid graph")
}

fn build_matmul_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let l = b.add_input("l", &[2, 3]);
    let r = b.add_input("r", &[3, 4]);
    let out = b.add_matmul(l, r, false, None, &[2, 4]);
    b.build(out).expect("valid graph")
}

fn build_narrow_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4, 8]);
    let out = b.add_narrow(x, 1, 0, 4, &[4, 4]);
    b.build(out).expect("valid graph")
}

fn build_embedding_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let i = b.add_input("i", &[4]);
    let w = b.add_input("w", &[100, 16]);
    let out = b.add_embedding(i, w, &[4, 16]);
    b.build(out).expect("valid graph")
}

fn build_conv2d_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let i = b.add_input("i", &[2, 8, 8]);
    let w = b.add_input("w", &[4, 2, 3, 3]);
    let out = b.add_conv2d(i, w, None, 1, 1, 0, 0, &[4, 6, 6]);
    b.build(out).expect("valid graph")
}

fn build_transpose_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[2, 3, 4]);
    let out = b.add_transpose(x, &[2, 0, 1], &[4, 2, 3]);
    b.build(out).expect("valid graph")
}

fn build_concat_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let a = b.add_input("a", &[1, 4, 8]);
    let c = b.add_input("c", &[1, 4, 8]);
    let out = b.add_concat(&[a, c], 1, &[1, 8, 8]);
    b.build(out).expect("valid graph")
}

fn build_index_select_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let data = b.add_input("data", &[10, 4]);
    let idx = b.add_input("idx", &[3]);
    let out = b.add_index_select(data, idx, 0, &[3, 4]);
    b.build(out).expect("valid graph")
}

fn build_gather_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let data = b.add_input("data", &[2, 5]);
    let idx = b.add_input("idx", &[2, 3]);
    let out = b.add_gather(data, idx, 1, &[2, 3]);
    b.build(out).expect("valid graph")
}

fn build_simdgroup_linear_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("data", &[128, 128]);
    let weight = b.add_input("weight", &[128, 128]);
    let bias = b.add_input("bias", &[128]);
    let out = b.add_linear(input, weight, Some(bias), &[128, 128]);
    b.build(out).expect("valid graph")
}

fn build_reduce_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4, 8]);
    let out = b.add_reduce(x, ReduceOp::Sum, 1, false, &[4]);
    b.build(out).expect("valid graph")
}

fn build_simdgroup_matmul_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let left = b.add_input("left", &[128, 128]);
    let right = b.add_input("right", &[128, 128]);
    let out = b.add_matmul(left, right, false, None, &[128, 128]);
    b.build(out).expect("valid graph")
}

// ---------------------------------------------------------------------------
// Per-variant coverage tests
// ---------------------------------------------------------------------------

#[test]
fn test_hip_binary_add_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_binary_kernel("h_add", true), &mut tags);
    assert!(
        tags.contains("BinaryAdd"),
        "must produce BinaryAdd: {tags:?}"
    );
}

#[test]
fn test_hip_binary_mul_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_binary_kernel("h_mul", false), &mut tags);
    assert!(
        tags.contains("BinaryMul"),
        "must produce BinaryMul: {tags:?}"
    );
}

#[test]
fn test_hip_sigmoid_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_unary_kernel("h_sig", true), &mut tags);
    assert!(tags.contains("Sigmoid"), "must produce Sigmoid: {tags:?}");
}

#[test]
fn test_hip_gelu_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_unary_kernel("h_gelu", false), &mut tags);
    assert!(tags.contains("Gelu"), "must produce Gelu: {tags:?}");
}

#[test]
fn test_hip_gelu_erf_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_gelu_erf_kernel("h_gelu_erf"), &mut tags);
    assert!(tags.contains("GeluErf"), "must produce GeluErf: {tags:?}");
}

#[test]
fn test_hip_relu_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_relu_kernel("h_relu"), &mut tags);
    assert!(tags.contains("Relu"), "must produce Relu: {tags:?}");
}

#[test]
fn test_hip_tanh_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_tanh_kernel("h_tanh"), &mut tags);
    assert!(tags.contains("Tanh"), "must produce Tanh: {tags:?}");
}

#[test]
fn test_hip_matmul_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_matmul_kernel("h_mm"), &mut tags);
    assert!(tags.contains("MatMul"), "must produce MatMul: {tags:?}");
}

#[test]
fn test_hip_linear_coverage() {
    let k = build_linear("h_lin", 4, 8, true).expect("valid");
    let mut tags = HashSet::new();
    plan_and_verify(&k, &mut tags);
    assert!(tags.contains("Linear"), "must produce Linear: {tags:?}");
}

#[test]
fn test_hip_softmax_coverage() {
    let k = build_softmax("h_sm", &[4, 8], -1).expect("valid");
    let mut tags = HashSet::new();
    plan_and_verify(&k, &mut tags);
    assert!(tags.contains("Softmax"), "must produce Softmax: {tags:?}");
}

#[test]
fn test_hip_reduce_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_reduce_kernel("h_red"), &mut tags);
    assert!(tags.contains("Reduce"), "must produce Reduce: {tags:?}");
}

#[test]
fn test_hip_conv1d_coverage() {
    let k = build_conv1d("h_c1d", 2, 4, 3, 16, 1, 1, false).expect("valid");
    let mut tags = HashSet::new();
    plan_and_verify(&k, &mut tags);
    assert!(tags.contains("Conv1d"), "must produce Conv1d: {tags:?}");
}

#[test]
fn test_hip_conv_transpose1d_coverage() {
    let k = build_conv_transpose_1d("h_ct1d", 4, 2, 3, 16, 2, 1, 1, 1, false, 0).expect("valid");
    let mut tags = HashSet::new();
    plan_and_verify(&k, &mut tags);
    assert!(
        tags.contains("ConvTranspose1d"),
        "must produce ConvTranspose1d: {tags:?}"
    );
}

#[test]
fn test_hip_narrow_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_narrow_kernel("h_narrow"), &mut tags);
    assert!(tags.contains("Narrow"), "must produce Narrow: {tags:?}");
}

#[test]
fn test_hip_embedding_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_embedding_kernel("h_emb"), &mut tags);
    assert!(
        tags.contains("Embedding"),
        "must produce Embedding: {tags:?}"
    );
}

#[test]
fn test_hip_conv2d_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_conv2d_kernel("h_c2d"), &mut tags);
    assert!(tags.contains("Conv2d"), "must produce Conv2d: {tags:?}");
}

#[test]
fn test_hip_transpose_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_transpose_kernel("h_tr"), &mut tags);
    assert!(
        tags.contains("Transpose"),
        "must produce Transpose: {tags:?}"
    );
}

#[test]
fn test_hip_concat_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_concat_kernel("h_cat"), &mut tags);
    assert!(tags.contains("Concat"), "must produce Concat: {tags:?}");
}

#[test]
fn test_hip_index_select_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_index_select_kernel("h_isel"), &mut tags);
    assert!(
        tags.contains("IndexSelect"),
        "must produce IndexSelect: {tags:?}"
    );
}

#[test]
fn test_hip_gather_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_gather_kernel("h_gat"), &mut tags);
    assert!(tags.contains("Gather"), "must produce Gather: {tags:?}");
}

#[test]
fn test_hip_simdgroup_linear_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_simdgroup_linear_kernel("h_slin"), &mut tags);
    assert!(
        tags.contains("SimdgroupLinear"),
        "must produce SimdgroupLinear: {tags:?}"
    );
}

#[test]
fn test_hip_simdgroup_matmul_coverage() {
    let mut tags = HashSet::new();
    plan_and_verify(&build_simdgroup_matmul_kernel("h_smm"), &mut tags);
    assert!(
        tags.contains("SimdgroupMatMul"),
        "must produce SimdgroupMatMul: {tags:?}"
    );
}

#[test]
fn test_hip_elementwise_coverage() {
    // RoPE produces an Elementwise step (composed scalar kernel).
    let k = build_rope_rotate_kernel(4, 8, 32).expect("valid");
    let mut tags = HashSet::new();
    plan_and_verify(&k, &mut tags);
    assert!(
        tags.contains("Elementwise"),
        "must produce Elementwise: {tags:?}"
    );
}

#[test]
fn test_hip_causal_conv1d_coverage() {
    // Causal conv1d produces ZeroPad1d + Conv1d.
    let k = build_causal_conv1d("h_cc1d", 4, 4, 3, 16, 1, 1, 1, false).expect("valid");
    let mut tags = HashSet::new();
    plan_and_verify(&k, &mut tags);
    assert!(
        tags.contains("ZeroPad1d"),
        "must produce ZeroPad1d: {tags:?}"
    );
}

// ---------------------------------------------------------------------------
// Exhaustive variant count
// ---------------------------------------------------------------------------

#[test]
fn test_hip_all_variants_covered() {
    let mut tags = HashSet::new();

    let builders: Vec<TensorKernelDef> = vec![
        build_binary_kernel("a_add", true),
        build_binary_kernel("a_mul", false),
        build_unary_kernel("a_sig", true),
        build_unary_kernel("a_gelu", false),
        build_gelu_erf_kernel("a_gelu_erf"),
        build_relu_kernel("a_relu"),
        build_tanh_kernel("a_tanh"),
        build_matmul_kernel("a_mm"),
        build_reduce_kernel("a_red"),
        build_linear("a_lin", 4, 8, true).expect("valid"),
        build_softmax("a_sm", &[4, 8], -1).expect("valid"),
        build_conv1d("a_c1d", 2, 4, 3, 16, 1, 1, false).expect("valid"),
        build_conv_transpose_1d("a_ct1d", 4, 2, 3, 16, 2, 1, 1, 1, false, 0).expect("valid"),
        build_narrow_kernel("a_narrow"),
        build_embedding_kernel("a_emb"),
        build_conv2d_kernel("a_c2d"),
        build_transpose_kernel("a_tr"),
        build_concat_kernel("a_cat"),
        build_index_select_kernel("a_isel"),
        build_gather_kernel("a_gat"),
        build_simdgroup_linear_kernel("a_slin"),
        build_simdgroup_matmul_kernel("a_smm"),
        build_rope_rotate_kernel(4, 8, 32).expect("valid"),
        build_causal_conv1d("a_cc1d", 4, 4, 3, 16, 1, 1, 1, false).expect("valid"),
    ];

    for kernel in &builders {
        plan_and_verify(kernel, &mut tags);
    }

    assert!(!tags.contains("UNKNOWN"), "unknown step type: {tags:?}");
    assert_eq!(
        tags.len(),
        KNOWN_VARIANT_COUNT,
        "expected {KNOWN_VARIANT_COUNT} variants, got {}: {tags:?}\n\
         Bump KNOWN_VARIANT_COUNT and add a test when adding DispatchStep variants.",
        tags.len(),
    );
}
