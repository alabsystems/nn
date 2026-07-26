// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: ViT FFN (feed-forward network) NY composition.
//!
//! Verifies bounds propagation through the full ViT MLP sub-block:
//!   input -> LayerNorm -> Linear(D, 4D) -> GELU -> Linear(4D, D) -> + residual
//!
//! Architecture (Dosovitskiy et al. 2020 "An Image is Worth 16x16 Words"):
//! - Pre-norm: LayerNorm before FFN (not post-norm)
//! - FFN expansion ratio: 4x (embed_dim -> 4*embed_dim -> embed_dim)
//! - GELU activation (not ReLU)
//! - Residual connection around the entire sub-block
//!
//! GELU is the key non-linearity that requires CROWN linearization.
//! LayerNorm requires heuristic linearization (IbpValidated mode).
//! Linear layers propagate exactly through both IBP and CROWN.
//!
//! Part of #3527: ViT encoder NY compose verification tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding, VerificationSoundnessMode};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions — small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Sequence length (number of patch tokens).
const SEQ_LEN: usize = 4;
/// Embedding dimension (tiny ViT hidden size).
const EMBED_DIM: usize = 64;
/// FFN intermediate dimension: 4x the embedding dimension per ViT spec.
const FFN_DIM: usize = 256;

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build a minimal ViT FFN kernel: Linear -> GELU -> Linear (no norm, no residual).
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable).
/// Output: `[SEQ_LEN, EMBED_DIM]`.
///
/// This isolates the MLP sub-block without normalization for clean CROWN testing.
fn build_vit_ffn_bare_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vit_ffn_bare");

    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let fc1_w = b.add_input("fc1_weight", &[FFN_DIM, EMBED_DIM]);
    let fc1_b = b.add_input("fc1_bias", &[FFN_DIM]);
    let fc2_w = b.add_input("fc2_weight", &[EMBED_DIM, FFN_DIM]);
    let fc2_b = b.add_input("fc2_bias", &[EMBED_DIM]);

    // Linear1: [S, D] -> [S, 4D]
    let h = b.add_linear(input, fc1_w, Some(fc1_b), &[SEQ_LEN, FFN_DIM]);
    // GELU activation: [S, 4D] -> [S, 4D]
    let h = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);
    // Linear2: [S, 4D] -> [S, D]
    let out = b.add_linear(h, fc2_w, Some(fc2_b), &[SEQ_LEN, EMBED_DIM]);

    b.build(out).expect("valid bare FFN kernel")
}

/// Build the full ViT FFN sub-block: LayerNorm -> Linear -> GELU -> Linear -> residual add.
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable).
/// Output: `[SEQ_LEN, EMBED_DIM]`.
///
/// This matches the pre-norm transformer FFN architecture used in ViT:
///   output = input + Linear2(GELU(Linear1(LayerNorm(input))))
fn build_vit_ffn_full_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vit_ffn_full");

    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[EMBED_DIM]);
    let ln_b = b.add_input("ln_bias", &[EMBED_DIM]);
    let fc1_w = b.add_input("fc1_weight", &[FFN_DIM, EMBED_DIM]);
    let fc1_b = b.add_input("fc1_bias", &[FFN_DIM]);
    let fc2_w = b.add_input("fc2_weight", &[EMBED_DIM, FFN_DIM]);
    let fc2_b = b.add_input("fc2_bias", &[EMBED_DIM]);

    // LayerNorm: [S, D] -> [S, D], normalizes along last axis (embed_dim)
    let normed = b.add_layer_norm(input, eps, 1, ln_w, ln_b, &[SEQ_LEN, EMBED_DIM]);
    // Linear1: [S, D] -> [S, 4D]
    let h = b.add_linear(normed, fc1_w, Some(fc1_b), &[SEQ_LEN, FFN_DIM]);
    // GELU activation: [S, 4D] -> [S, 4D]
    let h = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);
    // Linear2: [S, 4D] -> [S, D]
    let ffn_out = b.add_linear(h, fc2_w, Some(fc2_b), &[SEQ_LEN, EMBED_DIM]);
    // Residual: input + ffn_out
    let out = b.add_binary_add(input, ffn_out, &[SEQ_LEN, EMBED_DIM]);

    b.build(out).expect("valid full FFN kernel")
}

/// Bindings for the bare ViT FFN (no LayerNorm).
fn vit_ffn_bare_bindings() -> Vec<TensorParamBinding> {
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let fc1_b = ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32);
    let fc2_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM, FFN_DIM]), 0.02f32);
    let fc2_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,              // x [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantTensor(fc1_w), // fc1_weight [FFN_DIM, EMBED_DIM]
        TensorParamBinding::ConstantTensor(fc1_b), // fc1_bias [FFN_DIM]
        TensorParamBinding::ConstantTensor(fc2_w), // fc2_weight [EMBED_DIM, FFN_DIM]
        TensorParamBinding::ConstantTensor(fc2_b), // fc2_bias [EMBED_DIM]
    ]
}

/// Bindings for the full ViT FFN (with LayerNorm + residual).
fn vit_ffn_full_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let fc1_b = ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32);
    let fc2_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM, FFN_DIM]), 0.02f32);
    let fc2_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,              // x [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantScalar(1e-5),  // eps [1]
        TensorParamBinding::ConstantTensor(ln_w),  // ln_weight [EMBED_DIM]
        TensorParamBinding::ConstantTensor(ln_b),  // ln_bias [EMBED_DIM]
        TensorParamBinding::ConstantTensor(fc1_w), // fc1_weight [FFN_DIM, EMBED_DIM]
        TensorParamBinding::ConstantTensor(fc1_b), // fc1_bias [FFN_DIM]
        TensorParamBinding::ConstantTensor(fc2_w), // fc2_weight [EMBED_DIM, FFN_DIM]
        TensorParamBinding::ConstantTensor(fc2_b), // fc2_bias [EMBED_DIM]
    ]
}

// ---------------------------------------------------------------------------
// Bare FFN tests (Linear -> GELU -> Linear)
// ---------------------------------------------------------------------------

/// Bare FFN TensorKernelDef validates.
#[test]
fn test_vit_ffn_bare_def_validates() {
    let def = build_vit_ffn_bare_kernel();
    def.validate().expect("bare FFN kernel should validate");
}

/// Bare FFN translates to NY GraphNetwork.
#[test]
fn test_vit_ffn_bare_graph_builds() {
    let def = build_vit_ffn_bare_kernel();
    let bindings = vit_ffn_bare_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("bare FFN graph should translate");

    // Linear + GELU + Linear = at least 3 nodes.
    assert!(
        graph.num_nodes() >= 3,
        "bare FFN graph should have >= 3 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through bare ViT FFN.
#[test]
fn test_vit_ffn_bare_ibp_propagates() {
    let def = build_vit_ffn_bare_kernel();
    let bindings = vit_ffn_bare_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through bare FFN");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT FFN bare IBP: bounds=[{lo_min}, {hi_max}]");

    // Two linear layers with 0.02 weights + GELU. With small weights and
    // [-1, 1] input, output should be bounded.
    assert!(
        lo_min > -100.0,
        "IBP lower should be > -100 with small weights, got {lo_min}"
    );
    assert!(
        hi_max < 100.0,
        "IBP upper should be < 100 with small weights, got {hi_max}"
    );
}

/// CROWN bounds propagate through bare ViT FFN.
///
/// GELU is piecewise-smooth, so CROWN can linearize it. The linear layers
/// propagate exactly. CROWN should produce tighter bounds than IBP.
#[test]
fn test_vit_ffn_bare_crown_propagation() {
    let def = build_vit_ffn_bare_kernel();
    let bindings = vit_ffn_bare_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT FFN bare: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ---------------------------------------------------------------------------
// Full FFN tests (LayerNorm -> Linear -> GELU -> Linear -> residual)
// ---------------------------------------------------------------------------

/// Full FFN with LayerNorm and residual validates.
#[test]
fn test_vit_ffn_full_def_validates() {
    let def = build_vit_ffn_full_kernel();
    def.validate().expect("full FFN kernel should validate");
}

/// Full FFN translates to NY GraphNetwork.
#[test]
fn test_vit_ffn_full_graph_builds() {
    let def = build_vit_ffn_full_kernel();
    let bindings = vit_ffn_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("full FFN graph should translate");

    // LayerNorm + Linear + GELU + Linear + BinaryAdd = at least 5 nodes.
    assert!(
        graph.num_nodes() >= 5,
        "full FFN graph should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through full ViT FFN (with LayerNorm and residual).
#[test]
fn test_vit_ffn_full_ibp_propagates() {
    let def = build_vit_ffn_full_kernel();
    let bindings = vit_ffn_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through full FFN");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT FFN full IBP: bounds=[{lo_min}, {hi_max}]");

    // Residual adds input back, so bounds are at least as wide as input.
    // But with small weights the FFN branch contributes little.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

/// CROWN bounds propagate through full ViT FFN.
///
/// LayerNorm requires heuristic CROWN linearization (IbpValidated mode).
/// GELU linearizes cleanly. The residual adds the skip connection.
#[test]
fn test_vit_ffn_full_crown_propagation() {
    let def = build_vit_ffn_full_kernel();
    let bindings = vit_ffn_full_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ViT FFN full: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// Full FFN verify and record under "vit_ffn" key.
///
/// LayerNorm causes heuristic normalization approximation, so soundness
/// mode should be Heuristic (not Sound).
#[test]
fn test_vit_ffn_verify_and_record() {
    let def = build_vit_ffn_full_kernel();
    let bindings = vit_ffn_full_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "vit_ffn");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, EMBED_DIM]);

    // LayerNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "ViT FFN with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
