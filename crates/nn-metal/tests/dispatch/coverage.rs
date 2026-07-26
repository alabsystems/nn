// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Dispatch coverage test — verifies every `DispatchStep` variant has a handler
//! in `execute_tensor_dispatch`.
//!
//! Constructs kernels producing each `DispatchStep` variant and verifies via
//! `build_dispatch_plan()`. When a new variant is added, `KNOWN_VARIANT_COUNT`
//! fails, forcing the developer to add a test case and match arm.
//!
//! Part of #754.

use std::collections::HashSet;

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::{
    build_causal_conv1d, build_conv1d, build_conv_transpose_1d, build_dispatch_plan, build_linear,
    build_rope_rotate_kernel, build_softmax, DispatchStep, ScalarType, TensorKernelDef,
};

/// Number of `DispatchStep` variants. Update when adding new variants.
///
/// Reduce, Elementwise, Broadcast, Conv1d, ConvTranspose1d, Linear, MatMul,
/// BinaryAdd, BinaryMul, Sigmoid, Gelu, GeluErf, Relu, Tanh, LeakyRelu, Elu,
/// Exp, Softplus, Reshape, AxisSelect, Stack, Narrow, Softmax, ZeroPad1d,
/// Embedding, Conv2d, Transpose, Concat, IndexSelect, Gather,
/// SimdgroupLinear, SimdgroupMatMul, TiledLinear, TiledMatMul
const KNOWN_VARIANT_COUNT: usize = 34;

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
        DispatchStep::Tanh { .. } => "Tanh",
        DispatchStep::LeakyRelu { .. } => "LeakyRelu",
        DispatchStep::Elu { .. } => "Elu",
        DispatchStep::Exp { .. } => "Exp",
        DispatchStep::Softplus { .. } => "Softplus",
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

/// Collect unique step tags from a dispatch plan.
fn collect_tags(plan: &[DispatchStep]) -> Vec<&'static str> {
    let mut tags: Vec<_> = plan.iter().map(step_tag).collect();
    tags.sort_unstable();
    tags.dedup();
    tags
}

/// Build plan and collect tags into the accumulator set.
fn plan_tags(kernel: &TensorKernelDef, tags: &mut HashSet<&'static str>) {
    let (plan, _) = build_dispatch_plan(kernel, ScalarType::F32).expect("plan");
    for step in &plan {
        tags.insert(step_tag(step));
    }
}

/// Assert a kernel's plan contains the expected tag.
fn assert_tag(kernel: &TensorKernelDef, expected: &str) {
    let (plan, _) = build_dispatch_plan(kernel, ScalarType::F32).expect("plan");
    let tags = collect_tags(&plan);
    assert!(
        tags.contains(&expected),
        "must produce {expected}: {tags:?}"
    );
}

/// Build a simple binary op kernel (BinaryAdd or BinaryMul).
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

/// Build a simple unary activation kernel.
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

/// Build a gelu_erf activation kernel (exact erf path).
fn build_gelu_erf_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4]);
    let out = b.add_gelu_erf(x, &[4]);
    b.build(out).expect("valid graph")
}

/// Build a relu activation kernel.
fn build_relu_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4]);
    let out = b.add_relu(x, &[4]);
    b.build(out).expect("valid graph")
}

/// Build a tanh activation kernel.
fn build_tanh_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4]);
    let out = b.add_tanh(x, &[4]);
    b.build(out).expect("valid graph")
}

/// Build a leaky_relu activation kernel.
fn build_leaky_relu_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4]);
    let out = b.add_leaky_relu(x, 0.01, &[4]);
    b.build(out).expect("valid graph")
}

/// Build an elu activation kernel.
fn build_elu_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4]);
    let out = b.add_elu(x, 1.0, &[4]);
    b.build(out).expect("valid graph")
}

/// Build an exp activation kernel.
fn build_exp_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4]);
    let out = b.add_exp(x, &[4]);
    b.build(out).expect("valid graph")
}

/// Build a softplus activation kernel.
fn build_softplus_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4]);
    let out = b.add_softplus(x, &[4]);
    b.build(out).expect("valid graph")
}

/// Build a matmul kernel.
fn build_matmul_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let l = b.add_input("l", &[2, 3]);
    let r = b.add_input("r", &[3, 4]);
    let out = b.add_matmul(l, r, false, None, &[2, 4]);
    b.build(out).expect("valid graph")
}

/// Build a narrow kernel.
fn build_narrow_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[4, 8]);
    let out = b.add_narrow(x, 1, 0, 4, &[4, 4]);
    b.build(out).expect("valid graph")
}

/// Build an embedding kernel.
fn build_embedding_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let i = b.add_input("i", &[4]);
    let w = b.add_input("w", &[100, 16]);
    let out = b.add_embedding(i, w, &[4, 16]);
    b.build(out).expect("valid graph")
}

/// Build a conv2d kernel.
fn build_conv2d_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let i = b.add_input("i", &[2, 8, 8]);
    let w = b.add_input("w", &[4, 2, 3, 3]);
    let out = b.add_conv2d(i, w, None, 1, 1, 0, 0, &[4, 6, 6]);
    b.build(out).expect("valid graph")
}

/// Build a transpose kernel.
fn build_transpose_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let x = b.add_input("x", &[2, 3, 4]);
    let out = b.add_transpose(x, &[2, 0, 1], &[4, 2, 3]);
    b.build(out).expect("valid graph")
}

/// Build a concat kernel.
fn build_concat_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let a = b.add_input("a", &[1, 4, 8]);
    let c = b.add_input("c", &[1, 4, 8]);
    let out = b.add_concat(&[a, c], 1, &[1, 8, 8]);
    b.build(out).expect("valid graph")
}

/// Build an index_select kernel: select along dim 0 from [10, 4] with [3] indices → [3, 4].
fn build_index_select_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let data = b.add_input("data", &[10, 4]);
    let idx = b.add_input("idx", &[3]);
    let out = b.add_index_select(data, idx, 0, &[3, 4]);
    b.build(out).expect("valid graph")
}

/// Build a gather kernel: gather along dim 1 from [2, 5] with [2, 3] indices → [2, 3].
fn build_gather_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let data = b.add_input("data", &[2, 5]);
    let idx = b.add_input("idx", &[2, 3]);
    let out = b.add_gather(data, idx, 1, &[2, 3]);
    b.build(out).expect("valid graph")
}

/// Build a simdgroup-qualifying linear kernel.
///
/// Uses batched input [M, K] × weight [N, K] → [M, N] where M=128, K=128, N=128.
/// M×N = 16384, all dims % 8 == 0, K = 128 — meets simdgroup routing criteria.
fn build_simdgroup_linear_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("data", &[128, 128]);
    let weight = b.add_input("weight", &[128, 128]);
    let bias = b.add_input("bias", &[128]);
    let out = b.add_linear(input, weight, Some(bias), &[128, 128]);
    b.build(out).expect("valid graph")
}

/// Build a simdgroup-qualifying matmul kernel.
///
/// Left [128, 128] × Right [128, 128] → [128, 128].
/// M=128, K=128, N=128: M×N = 16384, all % 8, K ≥ 128.
fn build_simdgroup_matmul_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let left = b.add_input("left", &[128, 128]);
    let right = b.add_input("right", &[128, 128]);
    let out = b.add_matmul(left, right, false, None, &[128, 128]);
    b.build(out).expect("valid graph")
}

/// Build a tiled-qualifying linear kernel (below simdgroup threshold).
///
/// Input [32, 16] × Weight [32, 16] → [32, 32]. M=32, K=16, N=32.
/// Meets tiled (m>=16, k>=8, n>=16) but NOT simdgroup (M*N=1024 < 16384).
fn build_tiled_linear_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("data", &[32, 16]);
    let weight = b.add_input("weight", &[32, 16]);
    let bias = b.add_input("bias", &[32]);
    let out = b.add_linear(input, weight, Some(bias), &[32, 32]);
    b.build(out).expect("valid graph")
}

/// Build a tiled-qualifying matmul kernel (below simdgroup threshold).
///
/// Left [32, 16] × Right [16, 32] → [32, 32]. M=32, K=16, N=32.
/// Meets tiled (m>=16, k>=8, n>=16) but NOT simdgroup (M*N=1024 < 16384).
fn build_tiled_matmul_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let left = b.add_input("left", &[32, 16]);
    let right = b.add_input("right", &[16, 32]);
    let out = b.add_matmul(left, right, false, None, &[32, 32]);
    b.build(out).expect("valid graph")
}

// ---------------------------------------------------------------------------
// Per-variant coverage tests
// ---------------------------------------------------------------------------

#[test]
fn coverage_reduce_broadcast_elementwise() {
    // Decomposed instance norm produces Reduce + Broadcast + Elementwise.
    let kernel =
        nn_dsl::build_instance_norm_decomposed(1, 2, 4).expect("decomposed instance norm");
    let (plan, _) = build_dispatch_plan(&kernel, ScalarType::F32).expect("plan");
    let tags = collect_tags(&plan);
    assert!(tags.contains(&"Reduce"), "must produce Reduce: {tags:?}");
    assert!(
        tags.contains(&"Broadcast"),
        "must produce Broadcast: {tags:?}"
    );
    assert!(
        tags.contains(&"Elementwise"),
        "must produce Elementwise: {tags:?}"
    );
}

#[test]
fn coverage_reshape_axis_select_stack() {
    // RoPE produces Reshape, AxisSelect, Broadcast, Elementwise, Stack.
    let kernel = build_rope_rotate_kernel(1, 2, 4).expect("build RoPE");
    let (plan, _) = build_dispatch_plan(&kernel, ScalarType::F32).expect("plan");
    let tags = collect_tags(&plan);
    assert!(tags.contains(&"Reshape"), "RoPE: Reshape: {tags:?}");
    assert!(tags.contains(&"AxisSelect"), "RoPE: AxisSelect: {tags:?}");
    assert!(tags.contains(&"Stack"), "RoPE: Stack: {tags:?}");
}

#[test]
fn coverage_conv1d() {
    let kernel = build_conv1d("cov_conv1d", 2, 4, 3, 8, 1, 1, false).expect("build");
    assert_tag(&kernel, "Conv1d");
}

#[test]
fn coverage_conv_transpose_1d() {
    let kernel =
        build_conv_transpose_1d("cov_ct1d", 4, 2, 3, 4, 2, 0, 1, 1, false, 0).expect("build");
    assert_tag(&kernel, "ConvTranspose1d");
}

#[test]
fn coverage_linear() {
    let kernel = build_linear("cov_lin", 8, 4, true).expect("build");
    assert_tag(&kernel, "Linear");
}

#[test]
fn coverage_matmul() {
    assert_tag(&build_matmul_kernel("cov_mm"), "MatMul");
}

#[test]
fn coverage_binary_add() {
    assert_tag(&build_binary_kernel("cov_badd", true), "BinaryAdd");
}

#[test]
fn coverage_binary_mul() {
    assert_tag(&build_binary_kernel("cov_bmul", false), "BinaryMul");
}

#[test]
fn coverage_sigmoid() {
    assert_tag(&build_unary_kernel("cov_sig", true), "Sigmoid");
}

#[test]
fn coverage_gelu() {
    assert_tag(&build_unary_kernel("cov_gelu", false), "Gelu");
}

#[test]
fn coverage_gelu_erf() {
    assert_tag(&build_gelu_erf_kernel("cov_gelu_erf"), "GeluErf");
}

#[test]
fn coverage_relu() {
    assert_tag(&build_relu_kernel("cov_relu"), "Relu");
}

#[test]
fn coverage_tanh() {
    assert_tag(&build_tanh_kernel("cov_tanh"), "Tanh");
}

#[test]
fn coverage_leaky_relu() {
    assert_tag(&build_leaky_relu_kernel("cov_lrelu"), "LeakyRelu");
}

#[test]
fn coverage_elu() {
    assert_tag(&build_elu_kernel("cov_elu"), "Elu");
}

#[test]
fn coverage_exp() {
    assert_tag(&build_exp_kernel("cov_exp"), "Exp");
}

#[test]
fn coverage_softplus() {
    assert_tag(&build_softplus_kernel("cov_softplus"), "Softplus");
}

#[test]
fn coverage_narrow() {
    assert_tag(&build_narrow_kernel("cov_narrow"), "Narrow");
}

#[test]
fn coverage_softmax() {
    let kernel = build_softmax("cov_sm", &[2, 4], -1).expect("build");
    assert_tag(&kernel, "Softmax");
}

#[test]
fn coverage_zero_pad_1d() {
    let kernel = build_causal_conv1d("cov_zp", 2, 4, 3, 8, 1, 1, 1, false).expect("build");
    assert_tag(&kernel, "ZeroPad1d");
}

#[test]
fn coverage_embedding() {
    assert_tag(&build_embedding_kernel("cov_emb"), "Embedding");
}

#[test]
fn coverage_conv2d() {
    assert_tag(&build_conv2d_kernel("cov_conv2d"), "Conv2d");
}

#[test]
fn coverage_transpose() {
    assert_tag(&build_transpose_kernel("cov_transpose"), "Transpose");
}

#[test]
fn coverage_concat() {
    assert_tag(&build_concat_kernel("cov_concat"), "Concat");
}

#[test]
fn coverage_index_select() {
    assert_tag(&build_index_select_kernel("cov_isel"), "IndexSelect");
}

#[test]
fn coverage_gather() {
    assert_tag(&build_gather_kernel("cov_gather"), "Gather");
}

#[test]
fn coverage_simdgroup_linear() {
    assert_tag(
        &build_simdgroup_linear_kernel("cov_slin"),
        "SimdgroupLinear",
    );
}

#[test]
fn coverage_simdgroup_matmul() {
    assert_tag(&build_simdgroup_matmul_kernel("cov_smm"), "SimdgroupMatMul");
}

#[test]
fn coverage_tiled_linear() {
    assert_tag(&build_tiled_linear_kernel("cov_tlin"), "TiledLinear");
}

#[test]
fn coverage_tiled_matmul() {
    assert_tag(&build_tiled_matmul_kernel("cov_tmm"), "TiledMatMul");
}

// ---------------------------------------------------------------------------
// Aggregate: collects tags from all kernel types, asserts count == 34
// ---------------------------------------------------------------------------

#[test]
fn coverage_all_variants_exercised() {
    let mut tags: HashSet<&'static str> = HashSet::new();

    // Decomposed instance norm: Reduce, Broadcast, Elementwise
    plan_tags(
        &nn_dsl::build_instance_norm_decomposed(1, 2, 4).expect("inorm"),
        &mut tags,
    );
    // RoPE: Reshape, AxisSelect, Stack (plus Broadcast, Elementwise again)
    plan_tags(&build_rope_rotate_kernel(1, 2, 4).expect("rope"), &mut tags);
    // Conv1d, ConvTranspose1d, Linear, Softmax, ZeroPad1d+Conv1d
    plan_tags(
        &build_conv1d("a_c1d", 2, 4, 3, 8, 1, 1, false).expect("c"),
        &mut tags,
    );
    plan_tags(
        &build_conv_transpose_1d("a_ct", 4, 2, 3, 4, 2, 0, 1, 1, false, 0).expect("ct"),
        &mut tags,
    );
    plan_tags(&build_linear("a_lin", 8, 4, true).expect("l"), &mut tags);
    plan_tags(&build_softmax("a_sm", &[2, 4], -1).expect("s"), &mut tags);
    plan_tags(
        &build_causal_conv1d("a_zp", 2, 4, 3, 8, 1, 1, 1, false).expect("z"),
        &mut tags,
    );

    // Remaining variants via helpers
    let builders: Vec<TensorKernelDef> = vec![
        build_matmul_kernel("a_mm"),
        build_binary_kernel("a_ba", true),
        build_binary_kernel("a_bm", false),
        build_unary_kernel("a_sig", true),
        build_unary_kernel("a_gelu", false),
        build_gelu_erf_kernel("a_gelu_erf"),
        build_relu_kernel("a_relu"),
        build_tanh_kernel("a_tanh"),
        build_leaky_relu_kernel("a_lrelu"),
        build_elu_kernel("a_elu"),
        build_exp_kernel("a_exp"),
        build_softplus_kernel("a_softplus"),
        build_narrow_kernel("a_nar"),
        build_embedding_kernel("a_emb"),
        build_conv2d_kernel("a_conv2d"),
        build_transpose_kernel("a_transpose"),
        build_concat_kernel("a_concat"),
        build_index_select_kernel("a_isel"),
        build_gather_kernel("a_gather"),
        build_simdgroup_linear_kernel("a_slin"),
        build_simdgroup_matmul_kernel("a_smm"),
        build_tiled_linear_kernel("a_tlin"),
        build_tiled_matmul_kernel("a_tmm"),
    ];
    for kernel in &builders {
        plan_tags(kernel, &mut tags);
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
