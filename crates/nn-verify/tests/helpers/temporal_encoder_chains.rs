// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d -> normalization -> activation sub-chain tests for the HTDemucs
//! temporal encoder.
//!
//! Isolates the primitive composition patterns that appear inside the encoder:
//! Conv1d -> GroupNorm -> GELU, Conv1d -> InstanceNorm -> ReLU, etc. Small dims
//! (C<=16, T<=8) for NY tractability. Verifies that bounds propagate
//! correctly through each chain variant independently.
//!
//! Part of #3595 -- Compose verification for HTDemucs temporal encoder.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    uniform_bounds, verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// Small dims for chain tests (C<=16, T<=8).
const C_IN: usize = 4;
const C_OUT: usize = 8;
const T: usize = 8;
const KERNEL: usize = 3;
const STRIDE: usize = 1;
const PADDING: usize = 1;
const WEIGHT_MAG: f32 = 0.01;

/// Push a constant tensor binding.
fn push_weight(bindings: &mut Vec<TensorParamBinding>, shape: &[usize], val: f32) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(shape),
        val,
    )));
}

// ---------------------------------------------------------------------------
// Chain 1: Conv1d -> GroupNorm(G=1) -> GELU
//
// Core pattern inside DConv sub-layers: the compress path
// Conv1d(dilated) -> GN -> GELU before the expand Conv1d.
// ---------------------------------------------------------------------------

fn build_conv_gn_gelu() -> (nn_dsl::tensor_ir::TensorKernelDef, usize) {
    let mut b = TensorBlockBuilder::new("conv_gn_gelu_chain");
    let data = b.add_input("data", &[C_IN, T]);
    let cw = b.add_input("conv_w", &[C_OUT, C_IN, KERNEL]);
    let cb = b.add_input("conv_b", &[C_OUT]);
    let ng = b.add_input("norm_g", &[C_OUT]);
    let nb = b.add_input("norm_b", &[C_OUT]);
    let eps = b.add_input("eps", &[1]);

    let t_out = conv1d_out_len(T, KERNEL, STRIDE, PADDING);
    let x = b.add_conv1d(data, cw, Some(cb), STRIDE, PADDING, &[C_OUT, t_out]);
    let x = b.add_group_norm_g1(x, eps, Some(ng), Some(nb), C_OUT, t_out);
    let out = b.add_gelu(x, &[C_OUT, t_out]);

    (b.build(out).expect("valid conv_gn_gelu graph"), t_out)
}

fn conv_gn_gelu_bindings() -> Vec<TensorParamBinding> {
    let mut b = Vec::new();
    b.push(TensorParamBinding::Variable);
    push_weight(&mut b, &[C_OUT, C_IN, KERNEL], WEIGHT_MAG);
    push_weight(&mut b, &[C_OUT], 0.0);
    push_weight(&mut b, &[C_OUT], 1.0);
    push_weight(&mut b, &[C_OUT], 0.0);
    b.push(TensorParamBinding::ConstantScalar(1e-5));
    b
}

#[test]
fn test_conv_gn_gelu_ibp() {
    let (def, t_out) = build_conv_gn_gelu();
    let bindings = conv_gn_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[C_IN, T], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through conv_gn_gelu");
    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, t_out]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv->GN->GELU IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_conv_gn_gelu_crown() {
    let (def, t_out) = build_conv_gn_gelu();
    let bindings = conv_gn_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[C_IN, T], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, t_out]);
    assert_bounds_valid(&output);

    eprintln!("Conv->GN->GELU CROWN: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("  fallback: {reason}");
    }
}

// ---------------------------------------------------------------------------
// Chain 2: Conv1d -> InstanceNorm -> ReLU
//
// Alternate normalization + activation pattern. InstanceNorm is used in
// some encoder variants.
// ---------------------------------------------------------------------------

fn build_conv_instnorm_relu() -> (nn_dsl::tensor_ir::TensorKernelDef, usize) {
    let mut b = TensorBlockBuilder::new("conv_instnorm_relu_chain");
    let data = b.add_input("data", &[C_IN, T]);
    let cw = b.add_input("conv_w", &[C_OUT, C_IN, KERNEL]);
    let cb = b.add_input("conv_b", &[C_OUT]);
    let eps = b.add_input("eps", &[1]);

    let t_out = conv1d_out_len(T, KERNEL, STRIDE, PADDING);
    let x = b.add_conv1d(data, cw, Some(cb), STRIDE, PADDING, &[C_OUT, t_out]);
    let x = b.add_instance_norm(x, eps, 1, None, None, &[C_OUT, t_out]);
    let out = b.add_relu(x, &[C_OUT, t_out]);

    (b.build(out).expect("valid conv_instnorm_relu graph"), t_out)
}

fn conv_instnorm_relu_bindings() -> Vec<TensorParamBinding> {
    let mut b = Vec::new();
    b.push(TensorParamBinding::Variable);
    push_weight(&mut b, &[C_OUT, C_IN, KERNEL], WEIGHT_MAG);
    push_weight(&mut b, &[C_OUT], 0.0);
    b.push(TensorParamBinding::ConstantScalar(1e-5));
    b
}

#[test]
fn test_conv_instnorm_relu_ibp() {
    let (def, t_out) = build_conv_instnorm_relu();
    let bindings = conv_instnorm_relu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[C_IN, T], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through conv_instnorm_relu");
    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, t_out]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv->IN->ReLU IBP: [{lo_min}, {hi_max}]");
    // ReLU clamps lower bound >= 0.
    assert!(
        lo_min >= -1e-6,
        "ReLU output lower should be >= 0, got {lo_min}"
    );
}

#[test]
fn test_conv_instnorm_relu_crown() {
    let (def, t_out) = build_conv_instnorm_relu();
    let bindings = conv_instnorm_relu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[C_IN, T], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, t_out]);
    assert_bounds_valid(&output);

    eprintln!("Conv->IN->ReLU CROWN: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("  fallback: {reason}");
    }
}

// ---------------------------------------------------------------------------
// Chain 3: Conv1d(stride) -> GELU (no normalization)
//
// The entry point of the temporal encoder: Conv1d with stride downsampling
// followed by GELU. No normalization layer, so CROWN should succeed
// without fallback.
// ---------------------------------------------------------------------------

fn build_strided_conv_gelu() -> (nn_dsl::tensor_ir::TensorKernelDef, usize) {
    let stride = 2;
    let kernel = 4;
    let pad = 1;
    let mut b = TensorBlockBuilder::new("strided_conv_gelu_chain");
    let data = b.add_input("data", &[C_IN, T]);
    let cw = b.add_input("conv_w", &[C_OUT, C_IN, kernel]);
    let cb = b.add_input("conv_b", &[C_OUT]);

    let t_out = conv1d_out_len(T, kernel, stride, pad);
    let x = b.add_conv1d(data, cw, Some(cb), stride, pad, &[C_OUT, t_out]);
    let out = b.add_gelu(x, &[C_OUT, t_out]);

    (b.build(out).expect("valid strided_conv_gelu graph"), t_out)
}

fn strided_conv_gelu_bindings() -> Vec<TensorParamBinding> {
    let kernel = 4;
    let mut b = Vec::new();
    b.push(TensorParamBinding::Variable);
    push_weight(&mut b, &[C_OUT, C_IN, kernel], WEIGHT_MAG);
    push_weight(&mut b, &[C_OUT], 0.0);
    b
}

#[test]
fn test_strided_conv_gelu_ibp_and_shape() {
    let (def, t_out) = build_strided_conv_gelu();
    assert_eq!(t_out, 4, "Conv1d(k=4,s=2,p=1) on T=8 -> T=4");

    let bindings = strided_conv_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[C_IN, T], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through strided conv+gelu");
    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, t_out]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Strided Conv->GELU IBP: [{lo_min}, {hi_max}]");
    // GELU clamps most negative values near 0 -- lower bound should be small.
    assert!(
        lo_min > -1.0,
        "GELU output lower should be > -1.0, got {lo_min}"
    );
}

// ---------------------------------------------------------------------------
// Chain 4: Conv1d -> GroupNorm -> GLU
//
// The rewrite + GLU pattern at the encoder exit: Conv1d(k=1) doubles
// channels, then GLU halves them. Tests the multiplicative gating
// interaction through NY.
// ---------------------------------------------------------------------------

fn build_conv_gn_glu() -> (nn_dsl::tensor_ir::TensorKernelDef, usize) {
    let c_doubled = C_OUT * 2;
    let mut b = TensorBlockBuilder::new("conv_gn_glu_chain");
    let data = b.add_input("data", &[C_IN, T]);
    let cw = b.add_input("conv_w", &[c_doubled, C_IN, 1]);
    let cb = b.add_input("conv_b", &[c_doubled]);
    let ng = b.add_input("norm_g", &[c_doubled]);
    let nb = b.add_input("norm_b", &[c_doubled]);
    let eps = b.add_input("eps", &[1]);

    // Conv1d k=1: [C_IN, T] -> [c_doubled, T]
    let x = b.add_conv1d(data, cw, Some(cb), 1, 0, &[c_doubled, T]);
    let x = b.add_group_norm_g1(x, eps, Some(ng), Some(nb), c_doubled, T);
    // GLU halves along axis 0: [c_doubled, T] -> [C_OUT, T]
    let out = b.add_glu(x, 0, &[c_doubled, T]).expect("even channels");

    (b.build(out).expect("valid conv_gn_glu graph"), T)
}

fn conv_gn_glu_bindings() -> Vec<TensorParamBinding> {
    let c_doubled = C_OUT * 2;
    let mut b = Vec::new();
    b.push(TensorParamBinding::Variable);
    push_weight(&mut b, &[c_doubled, C_IN, 1], WEIGHT_MAG);
    push_weight(&mut b, &[c_doubled], 0.0);
    push_weight(&mut b, &[c_doubled], 1.0);
    push_weight(&mut b, &[c_doubled], 0.0);
    b.push(TensorParamBinding::ConstantScalar(1e-5));
    b
}

#[test]
fn test_conv_gn_glu_ibp() {
    let (def, t_out) = build_conv_gn_glu();
    let bindings = conv_gn_glu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[C_IN, T], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through conv_gn_glu");
    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, t_out]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv->GN->GLU IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_conv_gn_glu_crown() {
    let (def, t_out) = build_conv_gn_glu();
    let bindings = conv_gn_glu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[C_IN, T], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, t_out]);
    assert_bounds_valid(&output);

    eprintln!("Conv->GN->GLU CROWN: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("  fallback: {reason}");
    }
}

// ---------------------------------------------------------------------------
// Chain 5: Conv1d -> GELU -> Conv1d (two-conv chain, no norm)
//
// Tests sequential convolution composition without normalization.
// The encoder's entry Conv1d + GELU feeds into the DConv compress Conv1d.
// ---------------------------------------------------------------------------

fn build_conv_gelu_conv() -> nn_dsl::tensor_ir::TensorKernelDef {
    let c_mid = C_OUT;
    let c_final: usize = 16;
    let mut b = TensorBlockBuilder::new("conv_gelu_conv_chain");
    let data = b.add_input("data", &[C_IN, T]);
    let cw1 = b.add_input("conv1_w", &[c_mid, C_IN, KERNEL]);
    let cb1 = b.add_input("conv1_b", &[c_mid]);
    let cw2 = b.add_input("conv2_w", &[c_final, c_mid, KERNEL]);
    let cb2 = b.add_input("conv2_b", &[c_final]);

    let t1 = conv1d_out_len(T, KERNEL, STRIDE, PADDING);
    let x = b.add_conv1d(data, cw1, Some(cb1), STRIDE, PADDING, &[c_mid, t1]);
    let x = b.add_gelu(x, &[c_mid, t1]);
    let t2 = conv1d_out_len(t1, KERNEL, STRIDE, PADDING);
    let out = b.add_conv1d(x, cw2, Some(cb2), STRIDE, PADDING, &[c_final, t2]);

    b.build(out).expect("valid conv_gelu_conv graph")
}

fn conv_gelu_conv_bindings() -> Vec<TensorParamBinding> {
    let c_mid = C_OUT;
    let c_final: usize = 16;
    let mut b = Vec::new();
    b.push(TensorParamBinding::Variable);
    push_weight(&mut b, &[c_mid, C_IN, KERNEL], WEIGHT_MAG);
    push_weight(&mut b, &[c_mid], 0.0);
    push_weight(&mut b, &[c_final, c_mid, KERNEL], WEIGHT_MAG);
    push_weight(&mut b, &[c_final], 0.0);
    b
}

#[test]
fn test_conv_gelu_conv_ibp() {
    let def = build_conv_gelu_conv();
    let bindings = conv_gelu_conv_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[C_IN, T], 1.0);

    let t1 = conv1d_out_len(T, KERNEL, STRIDE, PADDING);
    let t2 = conv1d_out_len(t1, KERNEL, STRIDE, PADDING);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through conv_gelu_conv");
    assert_eq!(output.lower_upper().0.shape(), &[16, t2]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv->GELU->Conv IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

// ---------------------------------------------------------------------------
// Chain 6: Conv1d(dilated) -> GroupNorm -> GELU + residual skip
//
// Simplified single DConv iteration (no GLU/LayerScale). Tests dilated
// convolution + normalization + residual skip connection.
// ---------------------------------------------------------------------------

fn build_dilated_conv_gn_residual() -> nn_dsl::tensor_ir::TensorKernelDef {
    let c: usize = 8;
    let dilation: usize = 2;
    let dk: usize = 3;
    let dpad = dilation * (dk - 1) / 2;
    let mut b = TensorBlockBuilder::new("dilated_conv_gn_residual");
    let data = b.add_input("data", &[c, T]);
    let cw = b.add_input("dconv_w", &[c, c, dk]);
    let cb = b.add_input("dconv_b", &[c]);
    let ng = b.add_input("norm_g", &[c]);
    let nb = b.add_input("norm_b", &[c]);
    let eps = b.add_input("eps", &[1]);

    // Dilated Conv1d preserves T: pad = dilation * (K-1) / 2
    let x = b.add_conv1d_full(data, cw, Some(cb), 1, dpad, dilation, 1, &[c, T]);
    let x = b.add_group_norm_g1(x, eps, Some(ng), Some(nb), c, T);
    let x = b.add_gelu(x, &[c, T]);
    // Residual add
    let out = b.add_binary_add(data, x, &[c, T]);

    b.build(out).expect("valid dilated_conv_gn_residual graph")
}

fn dilated_conv_gn_residual_bindings() -> Vec<TensorParamBinding> {
    let c: usize = 8;
    let dk: usize = 3;
    let mut b = Vec::new();
    b.push(TensorParamBinding::Variable);
    push_weight(&mut b, &[c, c, dk], WEIGHT_MAG);
    push_weight(&mut b, &[c], 0.0);
    push_weight(&mut b, &[c], 1.0);
    push_weight(&mut b, &[c], 0.0);
    b.push(TensorParamBinding::ConstantScalar(1e-5));
    b
}

#[test]
fn test_dilated_conv_gn_residual_ibp() {
    let def = build_dilated_conv_gn_residual();
    let bindings = dilated_conv_gn_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[8, T], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dilated conv residual");
    // Output shape preserved by residual: [8, T].
    assert_eq!(output.lower_upper().0.shape(), &[8, T]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Dilated Conv->GN->GELU + residual IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite() && hi_max.is_finite());
}

#[test]
fn test_dilated_conv_gn_residual_crown() {
    let def = build_dilated_conv_gn_residual();
    let bindings = dilated_conv_gn_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[8, T], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[8, T]);
    assert_bounds_valid(&output);

    eprintln!("Dilated Conv->GN->GELU + residual CROWN: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("  fallback: {reason}");
    }
}

// ---------------------------------------------------------------------------
// Chain 7: verify_and_assert recording for Conv->GN->GELU chain
//
// Records the chain verification into the status file to integrate with
// the nn verification scorecard.
// ---------------------------------------------------------------------------

#[test]
fn test_conv_gn_gelu_verify_and_record() {
    let (def, t_out) = build_conv_gn_gelu();
    let bindings = conv_gn_gelu_bindings();
    let input = uniform_bounds(&[C_IN, T], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "demucs_temporal_encoder_conv_gn_gelu",
    );
    assert_eq!(result.num_variables, 1, "single Variable input");
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[C_OUT, t_out]
    );
}

// ---------------------------------------------------------------------------
// Chain 8: Conv1d -> GroupNorm -> Tanh
//
// Alternate activation (tanh). Tanh clamps output to [-1, 1], producing
// tighter bounds than GELU for the same input. Verifies NY
// handles the Tanh activation layer correctly.
// ---------------------------------------------------------------------------

fn build_conv_gn_tanh() -> (nn_dsl::tensor_ir::TensorKernelDef, usize) {
    let mut b = TensorBlockBuilder::new("conv_gn_tanh_chain");
    let data = b.add_input("data", &[C_IN, T]);
    let cw = b.add_input("conv_w", &[C_OUT, C_IN, KERNEL]);
    let cb = b.add_input("conv_b", &[C_OUT]);
    let ng = b.add_input("norm_g", &[C_OUT]);
    let nb = b.add_input("norm_b", &[C_OUT]);
    let eps = b.add_input("eps", &[1]);

    let t_out = conv1d_out_len(T, KERNEL, STRIDE, PADDING);
    let x = b.add_conv1d(data, cw, Some(cb), STRIDE, PADDING, &[C_OUT, t_out]);
    let x = b.add_group_norm_g1(x, eps, Some(ng), Some(nb), C_OUT, t_out);
    let out = b.add_tanh(x, &[C_OUT, t_out]);

    (b.build(out).expect("valid conv_gn_tanh graph"), t_out)
}

fn conv_gn_tanh_bindings() -> Vec<TensorParamBinding> {
    let mut b = Vec::new();
    b.push(TensorParamBinding::Variable);
    push_weight(&mut b, &[C_OUT, C_IN, KERNEL], WEIGHT_MAG);
    push_weight(&mut b, &[C_OUT], 0.0);
    push_weight(&mut b, &[C_OUT], 1.0);
    push_weight(&mut b, &[C_OUT], 0.0);
    b.push(TensorParamBinding::ConstantScalar(1e-5));
    b
}

#[test]
fn test_conv_gn_tanh_ibp() {
    let (def, t_out) = build_conv_gn_tanh();
    let bindings = conv_gn_tanh_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[C_IN, T], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through conv_gn_tanh");
    assert_eq!(output.lower_upper().0.shape(), &[C_OUT, t_out]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Conv->GN->Tanh IBP: [{lo_min}, {hi_max}]");
    // Tanh clamps output to [-1, 1].
    assert!(
        lo_min >= -1.0 - 1e-6,
        "tanh lower should be >= -1, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "tanh upper should be <= 1, got {hi_max}"
    );
}
