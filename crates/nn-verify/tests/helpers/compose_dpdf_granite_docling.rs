// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Granite-Docling vision-language model NY composition.
//!
//! Verifies bounds propagation through Granite-Docling sub-blocks used in the
//! dpdf document understanding pipeline:
//!
//! 1. **Patch embedding**: Conv2d(3, D, P, stride=P) -> reshape -> transpose
//!    (SigLIP2-style vision encoder front-end)
//!
//! 2. **RMSNorm**: Root mean square normalization used in Granite decoder layers
//!    instead of LayerNorm. RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps).
//!
//! 3. **SwiGLU FFN**: gate_proj -> SiLU -> mul(up_proj) -> down_proj
//!    Standard Granite/LLaMA-family FFN with gated linear units.
//!
//! 4. **Vision projection**: Linear projection mapping vision features to LM
//!    embedding space for cross-modal fusion.
//!
//! 5. **SigLIP2 MLP (GELU)**: Linear -> GELU -> Linear (vision encoder MLP).
//!
//! 6. **SigLIP2 multi-head attention**: Q/K/V projections -> scaled dot-product
//!    attention -> output projection. IBP and CROWN bounds.
//!
//! 7. **SigLIP2 encoder layer**: LayerNorm -> Attention -> residual -> LayerNorm
//!    -> MLP -> residual. Full pre-norm encoder layer. IBP and CROWN.
//!
//! 8. **SigLIP2 two-layer stack**: Two chained encoder layers testing CROWN depth.
//!
//! 9. **Granite GQA attention**: Grouped-query attention with 4:1 head ratio.
//!
//! 10. **Granite decoder layer**: RMSNorm -> Attention -> residual -> RMSNorm
//!     -> SwiGLU FFN -> residual. IBP and CROWN bounds.
//!
//! 11. **Granite RMSNorm -> SwiGLU composition**: Isolated normalization + gated
//!     FFN interaction with CROWN linearization.
//!
//! 12. **SigLIP2 to Granite projection**: Linear mapping vision features to LM
//!     embedding space.
//!
//! 13. **Full VLM compose**: End-to-end pipeline from image pixels through patch
//!     embedding, SigLIP2 encoder, vision projection, to Granite decoder FFN.
//!
//! Architecture references:
//! - Granite-Docling uses a SigLIP2-style vision encoder + Granite LLM decoder
//! - RMSNorm (Zhang & Sennrich, 2019) replaces LayerNorm in modern LLMs
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN used in LLaMA/Granite
//! - GQA (Ainslie et al., 2023): Grouped-Query Attention
//!
//! Dimensions (small for fast verification):
//! - IMG_SIZE=32, PATCH_SIZE=16, HIDDEN_DIM=64, FFN_DIM=128, SEQ_LEN=4
//!
//! Part of #3870, #3902: NY compose tests for dpdf models.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, ReduceOp, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Image height and width (square image).
const IMG_SIZE: usize = 32;
/// Patch size (P). IMG_SIZE must be divisible by PATCH_SIZE.
const PATCH_SIZE: usize = 16;
/// Number of patches per spatial dimension.
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 2
/// Total number of patches.
const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE; // 4
/// Input channels (RGB).
const IN_CHANNELS: usize = 3;
/// Hidden dimension (Granite decoder hidden size, tiny for testing).
const HIDDEN_DIM: usize = 64;
/// FFN intermediate dimension (SwiGLU uses separate gate and up projections).
const FFN_DIM: usize = 128;
/// Sequence length for decoder sub-block tests.
const SEQ_LEN: usize = 4;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// 1. Patch embedding: Conv2d -> reshape -> transpose
// ===========================================================================

/// Build a Granite-Docling patch embedding kernel using Conv2d.
///
/// Input: `[3, IMG_SIZE, IMG_SIZE]` (Variable, image pixels in [0, 1]).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]` after reshape and transpose.
///
/// Conv2d(in_channels=3, out_channels=D, kernel=P, stride=P, padding=0)
/// produces `[D, GRID_SIZE, GRID_SIZE]`.
/// Reshape to `[D, NUM_PATCHES]`, then transpose to `[NUM_PATCHES, D]`.
fn build_granite_patch_embedding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_docling_patch_embedding");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let weight = b.add_input(
        "proj_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let bias = b.add_input("proj_bias", &[HIDDEN_DIM]);

    // Conv2d: [3, 32, 32] -> [D, 2, 2]
    let conv_out = b.add_conv2d(
        input,
        weight,
        Some(bias),
        PATCH_SIZE, // stride_h
        PATCH_SIZE, // stride_w
        0,          // padding_h
        0,          // padding_w
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );

    // Reshape: [D, 2, 2] -> [D, NUM_PATCHES]
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);

    // Transpose: [D, NUM_PATCHES] -> [NUM_PATCHES, D]
    let out = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);

    b.build(out)
        .expect("valid Granite-Docling patch embedding kernel")
}

/// Create image-domain input bounds: pixels in [0, 1].
fn image_bounds_01(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Bindings for patch embedding.
fn granite_patch_embedding_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // image [3, 32, 32]
        TensorParamBinding::ConstantTensor(w),    // proj_weight [D, 3, P, P]
        TensorParamBinding::ConstantTensor(bias), // proj_bias [D]
    ]
}

/// Patch embedding TensorKernelDef validates.
#[test]
fn test_granite_docling_patch_embedding_def_validates() {
    let def = build_granite_patch_embedding_kernel();
    def.validate()
        .expect("Granite-Docling patch embedding kernel should validate");
}

/// Patch embedding translates to NY GraphNetwork.
#[test]
fn test_granite_docling_patch_embedding_graph_builds() {
    let def = build_granite_patch_embedding_kernel();
    let bindings = granite_patch_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("Granite-Docling patch embedding graph should translate");

    // Conv2d + Reshape + Transpose = at least 3 nodes
    assert!(
        graph.num_nodes() >= 3,
        "patch embedding graph should have >= 3 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through patch embedding with [0, 1] image input.
#[test]
fn test_granite_docling_patch_embedding_ibp_bounds() {
    let def = build_granite_patch_embedding_kernel();
    let bindings = granite_patch_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Granite-Docling patch embedding");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "output shape should be [NUM_PATCHES={NUM_PATCHES}, HIDDEN_DIM={HIDDEN_DIM}], got {:?}",
        lo.shape()
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite-Docling patch embedding IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

/// CROWN bounds propagate through patch embedding.
#[test]
fn test_granite_docling_patch_embedding_crown_propagation() {
    let def = build_granite_patch_embedding_kernel();
    let bindings = granite_patch_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite-Docling patch embedding: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record patch embedding.
#[test]
fn test_granite_docling_patch_embedding_verify_and_record() {
    let def = build_granite_patch_embedding_kernel();
    let bindings = granite_patch_embedding_bindings();
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let result = verify_and_assert(&def, &bindings, &input, "granite_docling_patch_embedding");
    assert_eq!(result.num_variables, 1, "single Variable input (image)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

// ===========================================================================
// 2. RMSNorm: RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
// ===========================================================================

/// Build an RMSNorm kernel for the Granite decoder.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, hidden states in [-1, 1]).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// RMSNorm normalizes by the root mean square of the input, without
/// subtracting the mean (unlike LayerNorm). This is computationally
/// cheaper and used in LLaMA/Granite/Qwen architectures.
fn build_granite_rmsnorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_docling_rmsnorm");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[HIDDEN_DIM]);

    // RMSNorm along the last axis (axis=1 for [SEQ_LEN, HIDDEN_DIM])
    let out = b.add_rms_norm(input, eps, 1, weight, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid Granite-Docling RMSNorm kernel")
}

/// Bindings for RMSNorm.
fn granite_rmsnorm_bindings() -> Vec<TensorParamBinding> {
    let weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);

    vec![
        TensorParamBinding::Variable,             // hidden [SEQ_LEN, HIDDEN_DIM]
        TensorParamBinding::ConstantScalar(1e-5), // eps [1]
        TensorParamBinding::ConstantTensor(weight), // weight [HIDDEN_DIM]
    ]
}

/// RMSNorm TensorKernelDef validates.
#[test]
fn test_granite_docling_rmsnorm_def_validates() {
    let def = build_granite_rmsnorm_kernel();
    def.validate()
        .expect("Granite-Docling RMSNorm kernel should validate");
}

/// RMSNorm translates to NY GraphNetwork.
#[test]
fn test_granite_docling_rmsnorm_graph_builds() {
    let def = build_granite_rmsnorm_kernel();
    let bindings = granite_rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("Granite-Docling RMSNorm graph should translate");

    assert!(
        graph.num_nodes() >= 1,
        "RMSNorm graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through RMSNorm with [-1, 1] hidden states.
///
/// RMSNorm with weight=1.0 normalizes the input to unit RMS. Output
/// bounds should be bounded (not vacuously wide).
#[test]
fn test_granite_docling_rmsnorm_ibp_bounds() {
    let def = build_granite_rmsnorm_kernel();
    let bindings = granite_rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Granite-Docling RMSNorm");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "RMSNorm output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite-Docling RMSNorm IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds propagate through RMSNorm.
///
/// RMSNorm involves division by sqrt(mean(x^2) + eps), which requires
/// CROWN linearization. Uses IbpValidated mode per nn engineering rules.
#[test]
fn test_granite_docling_rmsnorm_crown_propagation() {
    let def = build_granite_rmsnorm_kernel();
    let bindings = granite_rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite-Docling RMSNorm: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record RMSNorm.
#[test]
fn test_granite_docling_rmsnorm_verify_and_record() {
    let def = build_granite_rmsnorm_kernel();
    let bindings = granite_rmsnorm_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "granite_docling_rmsnorm");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 3. SwiGLU FFN: gate_proj -> SiLU -> mul(up_proj) -> down_proj
// ===========================================================================

/// Build a SwiGLU FFN kernel (Granite/LLaMA-style).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// SwiGLU FFN architecture (Shazeer 2020):
///   gate = gate_proj(x)                    [SEQ_LEN, FFN_DIM]
///   gate_activated = silu(gate)            [SEQ_LEN, FFN_DIM]
///   up = up_proj(x)                        [SEQ_LEN, FFN_DIM]
///   hidden = gate_activated * up           [SEQ_LEN, FFN_DIM]
///   output = down_proj(hidden)             [SEQ_LEN, HIDDEN_DIM]
///
/// SiLU = x * sigmoid(x), decomposed as sigmoid + binary_mul since
/// TensorBlockBuilder has no native add_silu.
fn build_granite_swiglu_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_docling_swiglu_ffn");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let gate_w = b.add_input("gate_proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_proj_weight", &[HIDDEN_DIM, FFN_DIM]);

    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // Gate branch: gate_proj -> SiLU
    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    // SiLU(x) = x * sigmoid(x): decomposed into sigmoid + binary_mul
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);

    // Up branch: up_proj
    let up = b.add_linear(input, up_w, None, &ffn_shape);

    // Multiplicative gating: silu(gate_proj(x)) * up_proj(x)
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);

    // Down projection: down_proj(hidden) -> [SEQ_LEN, HIDDEN_DIM]
    let out = b.add_linear(hidden, down_w, None, &out_shape);

    b.build(out)
        .expect("valid Granite-Docling SwiGLU FFN kernel")
}

/// Bindings for SwiGLU FFN.
fn granite_swiglu_ffn_bindings() -> Vec<TensorParamBinding> {
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,               // x [SEQ_LEN, HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(gate_w), // gate_proj_weight [FFN_DIM, HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(up_w),   // up_proj_weight [FFN_DIM, HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(down_w), // down_proj_weight [HIDDEN_DIM, FFN_DIM]
    ]
}

/// SwiGLU FFN TensorKernelDef validates.
#[test]
fn test_granite_docling_swiglu_ffn_def_validates() {
    let def = build_granite_swiglu_ffn_kernel();
    def.validate()
        .expect("Granite-Docling SwiGLU FFN kernel should validate");
}

/// SwiGLU FFN translates to NY GraphNetwork.
#[test]
fn test_granite_docling_swiglu_ffn_graph_builds() {
    let def = build_granite_swiglu_ffn_kernel();
    let bindings = granite_swiglu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("SwiGLU FFN graph should translate");

    // 3 Linear + 1 Sigmoid + 2 BinaryMul + 1 Linear = at least 7 nodes
    assert!(
        graph.num_nodes() >= 6,
        "SwiGLU FFN graph should have >= 6 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through SwiGLU FFN.
///
/// SwiGLU uses multiplicative gating: silu(gate_proj(x)) * up_proj(x).
/// The sigmoid in SiLU bounds gate_activated to [0, max], and the
/// multiplication with up_proj produces bounded output.
#[test]
fn test_granite_docling_swiglu_ffn_ibp_bounds() {
    let def = build_granite_swiglu_ffn_kernel();
    let bindings = granite_swiglu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Granite-Docling SwiGLU FFN");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "SwiGLU FFN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite-Docling SwiGLU FFN IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds propagate through SwiGLU FFN.
///
/// Sigmoid in SiLU is piecewise-smooth and can be linearized by CROWN.
/// The multiplicative gating produces a bilinear term that CROWN handles
/// via the McCormick envelope relaxation.
#[test]
fn test_granite_docling_swiglu_ffn_crown_propagation() {
    let def = build_granite_swiglu_ffn_kernel();
    let bindings = granite_swiglu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite-Docling SwiGLU FFN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record SwiGLU FFN.
#[test]
fn test_granite_docling_swiglu_ffn_verify_and_record() {
    let def = build_granite_swiglu_ffn_kernel();
    let bindings = granite_swiglu_ffn_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "granite_docling_swiglu_ffn");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 4. Vision projection: Linear mapping vision features to LM space
// ===========================================================================

/// Build a vision projection kernel.
///
/// Input: `[NUM_PATCHES, HIDDEN_DIM]` (Variable, vision encoder output).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]` (projected to LM embedding space).
///
/// In Granite-Docling, the vision projection maps SigLIP2 encoder output
/// to Granite LM hidden dimension. Here we use same-dim for simplicity;
/// the verification property (bounded linear projection) is the same
/// regardless of input/output dimension ratio.
fn build_granite_vision_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_docling_vision_projection");

    let input = b.add_input("vision_features", &[NUM_PATCHES, HIDDEN_DIM]);
    let weight = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("proj_bias", &[HIDDEN_DIM]);

    let out = b.add_linear(input, weight, Some(bias), &[NUM_PATCHES, HIDDEN_DIM]);

    b.build(out)
        .expect("valid Granite-Docling vision projection kernel")
}

/// Bindings for vision projection.
fn granite_vision_projection_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable, // vision_features [NUM_PATCHES, HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(w), // proj_weight [HIDDEN_DIM, HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(bias), // proj_bias [HIDDEN_DIM]
    ]
}

/// Vision projection TensorKernelDef validates.
#[test]
fn test_granite_docling_vision_projection_def_validates() {
    let def = build_granite_vision_projection_kernel();
    def.validate()
        .expect("vision projection kernel should validate");
}

/// Vision projection translates to NY GraphNetwork.
#[test]
fn test_granite_docling_vision_projection_graph_builds() {
    let def = build_granite_vision_projection_kernel();
    let bindings = granite_vision_projection_bindings();
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("vision projection graph should translate");

    assert!(
        graph.num_nodes() >= 1,
        "vision projection graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through vision projection.
///
/// Pure linear layer: output bounds scale with weight * input range.
/// With 0.02 weights, [-2, 2] input, D=64: max output ~= 0.02 * 64 * 2 = 2.56.
#[test]
fn test_granite_docling_vision_projection_ibp_bounds() {
    let def = build_granite_vision_projection_kernel();
    let bindings = granite_vision_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through vision projection");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "vision projection output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Granite-Docling vision projection IBP (features [-2,2]): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Linear with D=64, weight=0.02, input in [-2, 2]:
    // max output = sum(|w_i| * 2.0) = 64 * 0.02 * 2 = 2.56
    assert!(
        hi_max < 10.0,
        "vision projection upper should be < 10 with small weights, got {hi_max}"
    );
}

/// CROWN bounds propagate through vision projection (pure linear -- should succeed).
#[test]
fn test_granite_docling_vision_projection_crown_propagation() {
    let def = build_granite_vision_projection_kernel();
    let bindings = granite_vision_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 2.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_PATCHES, HIDDEN_DIM],);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite-Docling vision projection: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record vision projection.
#[test]
fn test_granite_docling_vision_projection_verify_and_record() {
    let def = build_granite_vision_projection_kernel();
    let bindings = granite_vision_projection_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 2.0);

    let result = verify_and_assert(&def, &bindings, &input, "granite_docling_vision_projection");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

// ===========================================================================
// 5. SigLIP2 MLP (GELU): Linear -> GELU -> Linear
// ===========================================================================

/// Build a SigLIP2 MLP block: Linear -> GELU -> Linear.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Standard transformer MLP with GELU activation used in the SigLIP2
/// vision encoder. Differs from the Granite SwiGLU FFN in activation
/// function (GELU vs SiLU-gated) and absence of multiplicative gating.
fn build_siglip2_mlp_gelu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_mlp_gelu");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let fc1_w = b.add_input("fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc1_b = b.add_input("fc1_bias", &[FFN_DIM]);
    let fc2_w = b.add_input("fc2_weight", &[HIDDEN_DIM, FFN_DIM]);
    let fc2_b = b.add_input("fc2_bias", &[HIDDEN_DIM]);

    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // Linear -> GELU -> Linear
    let h = b.add_linear(input, fc1_w, Some(fc1_b), &ffn_shape);
    let h = b.add_gelu(h, &ffn_shape);
    let out = b.add_linear(h, fc2_w, Some(fc2_b), &out_shape);

    b.build(out).expect("valid SigLIP2 MLP GELU kernel")
}

/// Bindings for SigLIP2 MLP GELU.
fn siglip2_mlp_gelu_bindings() -> Vec<TensorParamBinding> {
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_b = ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let fc2_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,              // x [SEQ_LEN, HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(fc1_w), // fc1_weight [FFN_DIM, HIDDEN_DIM]
        TensorParamBinding::ConstantTensor(fc1_b), // fc1_bias [FFN_DIM]
        TensorParamBinding::ConstantTensor(fc2_w), // fc2_weight [HIDDEN_DIM, FFN_DIM]
        TensorParamBinding::ConstantTensor(fc2_b), // fc2_bias [HIDDEN_DIM]
    ]
}

/// IBP bounds propagate through SigLIP2 MLP (Linear -> GELU -> Linear).
///
/// GELU is a smooth activation with bounded derivative. Output bounds
/// should be finite and non-degenerate.
#[test]
fn test_siglip2_mlp_gelu_ibp() {
    let def = build_siglip2_mlp_gelu_kernel();
    let bindings = siglip2_mlp_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through SigLIP2 MLP GELU");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "MLP GELU output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 MLP GELU IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. SigLIP2 Multi-Head Attention IBP/CROWN
// ===========================================================================

/// Number of attention heads for SigLIP2.
const SIGLIP2_NUM_HEADS: usize = 4;
/// Head dimension for SigLIP2 = HIDDEN_DIM / SIGLIP2_NUM_HEADS.
const SIGLIP2_HEAD_DIM: usize = HIDDEN_DIM / SIGLIP2_NUM_HEADS; // 16

/// Build a SigLIP2 multi-head attention kernel.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Q/K/V linear projections -> scaled dot-product attention (with softmax)
/// -> output projection. Standard multi-head self-attention as used in
/// SigLIP2 vision encoder (all heads have same K/V, no GQA).
fn build_siglip2_multi_head_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_multi_head_attention");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let q_b = b.add_input("q_proj_bias", &[HIDDEN_DIM]);
    let k_w = b.add_input("k_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_b = b.add_input("k_proj_bias", &[HIDDEN_DIM]);
    let v_w = b.add_input("v_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_b = b.add_input("v_proj_bias", &[HIDDEN_DIM]);
    let out_w = b.add_input("out_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_b = b.add_input("out_proj_bias", &[HIDDEN_DIM]);

    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Q/K/V projections
    let q = b.add_linear(input, q_w, Some(q_b), &shape);
    let k = b.add_linear(input, k_w, Some(k_b), &shape);
    let v = b.add_linear(input, v_w, Some(v_b), &shape);

    // Scaled dot-product attention with softmax
    let scale = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);

    // Output projection
    let out = b.add_linear(attn, out_w, Some(out_b), &shape);

    b.build(out)
        .expect("valid SigLIP2 multi-head attention kernel")
}

/// Bindings for SigLIP2 multi-head attention.
fn siglip2_multi_head_attention_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let q_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let k_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let v_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,              // hidden
        TensorParamBinding::ConstantTensor(q_w),   // q_proj_weight
        TensorParamBinding::ConstantTensor(q_b),   // q_proj_bias
        TensorParamBinding::ConstantTensor(k_w),   // k_proj_weight
        TensorParamBinding::ConstantTensor(k_b),   // k_proj_bias
        TensorParamBinding::ConstantTensor(v_w),   // v_proj_weight
        TensorParamBinding::ConstantTensor(v_b),   // v_proj_bias
        TensorParamBinding::ConstantTensor(out_w), // out_proj_weight
        TensorParamBinding::ConstantTensor(out_b), // out_proj_bias
    ]
}

/// IBP bounds propagate through SigLIP2 multi-head attention.
///
/// Q/K/V linear projections -> matmul -> scale -> softmax -> matmul -> output
/// projection. Softmax bounds are in [0, 1], so attention output is bounded
/// by V bounds. Output projection further scales by weight magnitude.
#[test]
fn test_siglip2_multi_head_attention_ibp() {
    let def = build_siglip2_multi_head_attention_kernel();
    let bindings = siglip2_multi_head_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through SigLIP2 multi-head attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "multi-head attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 multi-head attention IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds propagate through SigLIP2 multi-head attention.
///
/// CROWN linearizes the softmax non-linearity within the attention op,
/// producing tighter bounds than IBP for the Q@K^T -> softmax -> @V chain.
#[test]
fn test_siglip2_multi_head_attention_crown() {
    let def = build_siglip2_multi_head_attention_kernel();
    let bindings = siglip2_multi_head_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 multi-head attention: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 7. SigLIP2 Encoder Layer IBP/CROWN
// ===========================================================================

/// Build a SigLIP2 encoder layer:
/// LayerNorm -> Attention -> residual -> LayerNorm -> MLP(Linear->GELU->Linear) -> residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Standard pre-norm transformer encoder layer as used in SigLIP2/ViT.
fn build_siglip2_encoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_encoder_layer");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Pre-attention LayerNorm
    let ln1_eps = b.add_input("ln1_eps", &[1]);
    let ln1_w = b.add_input("ln1_weight", &[HIDDEN_DIM]);
    let ln1_b = b.add_input("ln1_bias", &[HIDDEN_DIM]);
    let normed1 = b.add_layer_norm(input, ln1_eps, 1, ln1_w, ln1_b, &shape);

    // Self-attention (Q/K/V projection + attention + output projection)
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);

    let scale = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // Residual connection after attention
    let residual1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN LayerNorm
    let ln2_eps = b.add_input("ln2_eps", &[1]);
    let ln2_w = b.add_input("ln2_weight", &[HIDDEN_DIM]);
    let ln2_b = b.add_input("ln2_bias", &[HIDDEN_DIM]);
    let normed2 = b.add_layer_norm(residual1, ln2_eps, 1, ln2_w, ln2_b, &shape);

    // MLP: Linear -> GELU -> Linear
    let fc1_w = b.add_input("fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc1_b = b.add_input("fc1_bias", &[FFN_DIM]);
    let fc2_w = b.add_input("fc2_weight", &[HIDDEN_DIM, FFN_DIM]);
    let fc2_b = b.add_input("fc2_bias", &[HIDDEN_DIM]);

    let h = b.add_linear(normed2, fc1_w, Some(fc1_b), &ffn_shape);
    let h = b.add_gelu(h, &ffn_shape);
    let ffn_out = b.add_linear(h, fc2_w, Some(fc2_b), &shape);

    // Residual connection after FFN
    let out = b.add_binary_add(residual1, ffn_out, &shape);

    b.build(out).expect("valid SigLIP2 encoder layer kernel")
}

/// Bindings for SigLIP2 encoder layer.
fn siglip2_encoder_layer_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_b = ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let fc2_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                     // hidden
        TensorParamBinding::ConstantScalar(1e-5),         // ln1_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // ln1_bias
        TensorParamBinding::ConstantTensor(q_w),          // q_weight
        TensorParamBinding::ConstantTensor(k_w),          // k_weight
        TensorParamBinding::ConstantTensor(v_w),          // v_weight
        TensorParamBinding::ConstantTensor(out_w),        // out_weight
        TensorParamBinding::ConstantScalar(1e-5),         // ln2_eps
        TensorParamBinding::ConstantTensor(ln_w),         // ln2_weight
        TensorParamBinding::ConstantTensor(ln_b),         // ln2_bias
        TensorParamBinding::ConstantTensor(fc1_w),        // fc1_weight
        TensorParamBinding::ConstantTensor(fc1_b),        // fc1_bias
        TensorParamBinding::ConstantTensor(fc2_w),        // fc2_weight
        TensorParamBinding::ConstantTensor(fc2_b),        // fc2_bias
    ]
}

/// IBP bounds propagate through full SigLIP2 encoder layer.
///
/// LayerNorm -> Attention -> residual -> LayerNorm -> MLP -> residual.
/// Residual connections preserve bounded output.
#[test]
fn test_siglip2_encoder_layer_ibp() {
    let def = build_siglip2_encoder_layer_kernel();
    let bindings = siglip2_encoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through SigLIP2 encoder layer");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "encoder layer output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 encoder layer IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds propagate through full SigLIP2 encoder layer.
///
/// Tests CROWN linearization through LayerNorm (normalization) and
/// softmax (attention). Uses IbpValidated mode per nn engineering rules.
#[test]
fn test_siglip2_encoder_layer_crown() {
    let def = build_siglip2_encoder_layer_kernel();
    let bindings = siglip2_encoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 encoder layer: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. SigLIP2 Two-Layer Stack IBP
// ===========================================================================

/// Build two chained SigLIP2 encoder layers.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Tests CROWN depth: bounds propagation through two consecutive
/// encoder layers (2x LayerNorm + 2x Attention + 2x MLP).
fn build_siglip2_two_layer_stack_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_two_layer_stack");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // ---- Layer 1 ----
    let ln1a_eps = b.add_input("l1_ln1_eps", &[1]);
    let ln1a_w = b.add_input("l1_ln1_weight", &[HIDDEN_DIM]);
    let ln1a_b = b.add_input("l1_ln1_bias", &[HIDDEN_DIM]);
    let normed1a = b.add_layer_norm(input, ln1a_eps, 1, ln1a_w, ln1a_b, &shape);

    let q1_w = b.add_input("l1_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k1_w = b.add_input("l1_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v1_w = b.add_input("l1_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out1_w = b.add_input("l1_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q1 = b.add_linear(normed1a, q1_w, None, &shape);
    let k1 = b.add_linear(normed1a, k1_w, None, &shape);
    let v1 = b.add_linear(normed1a, v1_w, None, &shape);

    let scale = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();
    let attn1 = b.add_attention(q1, k1, v1, AttentionMask::Standard, Some(scale), &shape);
    let attn1_out = b.add_linear(attn1, out1_w, None, &shape);
    let res1a = b.add_binary_add(input, attn1_out, &shape);

    let ln1b_eps = b.add_input("l1_ln2_eps", &[1]);
    let ln1b_w = b.add_input("l1_ln2_weight", &[HIDDEN_DIM]);
    let ln1b_b = b.add_input("l1_ln2_bias", &[HIDDEN_DIM]);
    let normed1b = b.add_layer_norm(res1a, ln1b_eps, 1, ln1b_w, ln1b_b, &shape);

    let fc1a_w = b.add_input("l1_fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc2a_w = b.add_input("l1_fc2_weight", &[HIDDEN_DIM, FFN_DIM]);

    let h1 = b.add_linear(normed1b, fc1a_w, None, &ffn_shape);
    let h1 = b.add_gelu(h1, &ffn_shape);
    let ffn1_out = b.add_linear(h1, fc2a_w, None, &shape);
    let layer1_out = b.add_binary_add(res1a, ffn1_out, &shape);

    // ---- Layer 2 ----
    let ln2a_eps = b.add_input("l2_ln1_eps", &[1]);
    let ln2a_w = b.add_input("l2_ln1_weight", &[HIDDEN_DIM]);
    let ln2a_b = b.add_input("l2_ln1_bias", &[HIDDEN_DIM]);
    let normed2a = b.add_layer_norm(layer1_out, ln2a_eps, 1, ln2a_w, ln2a_b, &shape);

    let q2_w = b.add_input("l2_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k2_w = b.add_input("l2_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v2_w = b.add_input("l2_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out2_w = b.add_input("l2_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q2 = b.add_linear(normed2a, q2_w, None, &shape);
    let k2 = b.add_linear(normed2a, k2_w, None, &shape);
    let v2 = b.add_linear(normed2a, v2_w, None, &shape);

    let attn2 = b.add_attention(q2, k2, v2, AttentionMask::Standard, Some(scale), &shape);
    let attn2_out = b.add_linear(attn2, out2_w, None, &shape);
    let res2a = b.add_binary_add(layer1_out, attn2_out, &shape);

    let ln2b_eps = b.add_input("l2_ln2_eps", &[1]);
    let ln2b_w = b.add_input("l2_ln2_weight", &[HIDDEN_DIM]);
    let ln2b_b = b.add_input("l2_ln2_bias", &[HIDDEN_DIM]);
    let normed2b = b.add_layer_norm(res2a, ln2b_eps, 1, ln2b_w, ln2b_b, &shape);

    let fc1b_w = b.add_input("l2_fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc2b_w = b.add_input("l2_fc2_weight", &[HIDDEN_DIM, FFN_DIM]);

    let h2 = b.add_linear(normed2b, fc1b_w, None, &ffn_shape);
    let h2 = b.add_gelu(h2, &ffn_shape);
    let ffn2_out = b.add_linear(h2, fc2b_w, None, &shape);
    let out = b.add_binary_add(res2a, ffn2_out, &shape);

    b.build(out).expect("valid SigLIP2 two-layer stack kernel")
}

/// Bindings for SigLIP2 two-layer stack.
fn siglip2_two_layer_stack_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    // Layer 1 + Layer 2: each has 2 LN (eps+weight+bias) + 4 attn weights + 2 FFN weights
    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    // Layer 1
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // l1_ln1_eps
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // l1_ln1_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // l1_ln1_bias
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l1_q_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l1_k_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l1_v_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l1_out_weight
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // l1_ln2_eps
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // l1_ln2_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // l1_ln2_bias
    bindings.push(TensorParamBinding::ConstantTensor(fc1_w.clone())); // l1_fc1_weight
    bindings.push(TensorParamBinding::ConstantTensor(fc2_w.clone())); // l1_fc2_weight

    // Layer 2
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // l2_ln1_eps
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // l2_ln1_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // l2_ln1_bias
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l2_q_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l2_k_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l2_v_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w)); // l2_out_weight
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // l2_ln2_eps
    bindings.push(TensorParamBinding::ConstantTensor(ln_w)); // l2_ln2_weight
    bindings.push(TensorParamBinding::ConstantTensor(ln_b)); // l2_ln2_bias
    bindings.push(TensorParamBinding::ConstantTensor(fc1_w)); // l2_fc1_weight
    bindings.push(TensorParamBinding::ConstantTensor(fc2_w)); // l2_fc2_weight

    bindings
}

/// IBP bounds propagate through two chained SigLIP2 encoder layers.
///
/// Tests depth-2 bound propagation: bounds should remain finite
/// through two sequential encoder layers with residual connections.
#[test]
fn test_siglip2_two_layer_stack_ibp() {
    let def = build_siglip2_two_layer_stack_kernel();
    let bindings = siglip2_two_layer_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through SigLIP2 two-layer stack");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "two-layer stack output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 two-layer stack IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. Granite GQA Attention IBP
// ===========================================================================

/// Number of query heads for Granite decoder.
const GRANITE_NUM_HEADS: usize = 4;
/// Number of KV heads for grouped-query attention (4:1 ratio).
const GRANITE_NUM_KV_HEADS: usize = 1;
/// Head dimension = HIDDEN_DIM / GRANITE_NUM_HEADS.
const GRANITE_HEAD_DIM: usize = HIDDEN_DIM / GRANITE_NUM_HEADS; // 16
/// KV dimension = GRANITE_NUM_KV_HEADS * GRANITE_HEAD_DIM.
const GRANITE_KV_DIM: usize = GRANITE_NUM_KV_HEADS * GRANITE_HEAD_DIM; // 16

/// Build a Granite grouped-query attention kernel.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// GQA with 4 query heads and 1 KV head (4:1 ratio). For verification
/// tractability, Q is projected down to KV_DIM for attention, then
/// projected back up to HIDDEN_DIM.
fn build_granite_gqa_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_gqa_attention");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // K/V projections: [SEQ_LEN, HIDDEN_DIM] -> [SEQ_LEN, GRANITE_KV_DIM]
    let k_w = b.add_input("k_proj_weight", &[GRANITE_KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_proj_weight", &[GRANITE_KV_DIM, HIDDEN_DIM]);

    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, GRANITE_KV_DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, GRANITE_KV_DIM]);

    // For GQA with 4:1 ratio, project Q down to KV_DIM for attention
    let q_down_w = b.add_input("q_down_weight", &[GRANITE_KV_DIM, HIDDEN_DIM]);
    let q_down = b.add_linear(input, q_down_w, None, &[SEQ_LEN, GRANITE_KV_DIM]);

    // Attention: Q_down @ K^T -> softmax -> @ V -> [SEQ_LEN, GRANITE_KV_DIM]
    let scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();
    let attn_out = b.add_attention(
        q_down,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, GRANITE_KV_DIM],
    );

    // Output projection: [SEQ_LEN, GRANITE_KV_DIM] -> [SEQ_LEN, HIDDEN_DIM]
    let out_up_w = b.add_input("out_up_weight", &[HIDDEN_DIM, GRANITE_KV_DIM]);
    let out = b.add_linear(attn_out, out_up_w, None, &shape);

    b.build(out).expect("valid Granite GQA attention kernel")
}

/// Bindings for Granite GQA attention.
fn granite_gqa_attention_bindings() -> Vec<TensorParamBinding> {
    let k_w = ArrayD::from_elem(IxDyn(&[GRANITE_KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[GRANITE_KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let q_down_w = ArrayD::from_elem(IxDyn(&[GRANITE_KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_up_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, GRANITE_KV_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                 // hidden
        TensorParamBinding::ConstantTensor(k_w),      // k_proj_weight
        TensorParamBinding::ConstantTensor(v_w),      // v_proj_weight
        TensorParamBinding::ConstantTensor(q_down_w), // q_down_weight
        TensorParamBinding::ConstantTensor(out_up_w), // out_up_weight
    ]
}

/// IBP bounds propagate through Granite GQA attention with 4:1 head ratio.
///
/// Grouped-query attention with NUM_KV_HEADS=1 < NUM_HEADS=4. Q is projected
/// to KV_DIM for attention, then back to HIDDEN_DIM. Causal mask ensures
/// position j attends only to positions <= j.
#[test]
fn test_granite_gqa_attention_ibp() {
    let def = build_granite_gqa_attention_kernel();
    let bindings = granite_gqa_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Granite GQA attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "Granite GQA attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite GQA attention IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. Granite Decoder Layer IBP/CROWN
// ===========================================================================

/// Build a Granite decoder layer:
/// RMSNorm -> GQA Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Granite decoder uses RMSNorm (not LayerNorm) and SwiGLU (not GELU MLP).
/// This is the standard LLaMA/Granite decoder layer architecture.
fn build_granite_decoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_decoder_layer");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Pre-attention RMSNorm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // Self-attention with full-dim Q/K/V for verification tractability
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);

    let scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // Residual connection after attention
    let residual1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(residual1, norm2_eps, 1, norm2_w, &shape);

    // SwiGLU FFN: gate_proj -> SiLU -> mul(up_proj) -> down_proj
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Residual connection after FFN
    let out = b.add_binary_add(residual1, ffn_out, &shape);

    b.build(out).expect("valid Granite decoder layer kernel")
}

/// Bindings for Granite decoder layer.
fn granite_decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(q_w),            // q_weight
        TensorParamBinding::ConstantTensor(k_w),            // k_weight
        TensorParamBinding::ConstantTensor(v_w),            // v_weight
        TensorParamBinding::ConstantTensor(out_w),          // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
    ]
}

/// IBP bounds propagate through full Granite decoder layer.
///
/// RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
#[test]
fn test_granite_decoder_layer_ibp() {
    let def = build_granite_decoder_layer_kernel();
    let bindings = granite_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Granite decoder layer");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "Granite decoder layer output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite decoder layer IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds propagate through full Granite decoder layer.
///
/// Tests CROWN linearization through RMSNorm (normalization), softmax
/// (attention), sigmoid (SiLU in SwiGLU), and binary multiplication
/// (McCormick envelope). Uses IbpValidated mode.
#[test]
fn test_granite_decoder_layer_crown() {
    let def = build_granite_decoder_layer_kernel();
    let bindings = granite_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite decoder layer: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 11. Granite RMSNorm -> SwiGLU Composition CROWN
// ===========================================================================

/// Build a Granite RMSNorm -> SwiGLU composition.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Tests tight CROWN bounds through RMSNorm -> SwiGLU specifically,
/// without the attention block. This isolates the normalization +
/// gated FFN interaction that is unique to LLaMA/Granite architectures.
fn build_granite_rmsnorm_swiglu_compose_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_rmsnorm_swiglu_compose");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // RMSNorm
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    // SwiGLU FFN
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &shape);

    b.build(out)
        .expect("valid Granite RMSNorm -> SwiGLU compose kernel")
}

/// Bindings for Granite RMSNorm -> SwiGLU composition.
fn granite_rmsnorm_swiglu_compose_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantScalar(1e-5),   // eps
        TensorParamBinding::ConstantTensor(norm_w), // norm_weight
        TensorParamBinding::ConstantTensor(gate_w), // gate_weight
        TensorParamBinding::ConstantTensor(up_w),   // up_weight
        TensorParamBinding::ConstantTensor(down_w), // down_weight
    ]
}

/// CROWN bounds through Granite RMSNorm -> SwiGLU composition.
///
/// Isolates the normalization + gated FFN interaction. CROWN linearizes
/// through RMSNorm (division by sqrt(mean(x^2) + eps)) and SiLU sigmoid,
/// then handles the McCormick envelope for multiplicative gating.
#[test]
fn test_granite_rmsnorm_swiglu_compose_crown() {
    let def = build_granite_rmsnorm_swiglu_compose_kernel();
    let bindings = granite_rmsnorm_swiglu_compose_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite RMSNorm->SwiGLU compose: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 12. SigLIP2 to Granite Projection IBP
// ===========================================================================

/// Projection dimension (Granite LM embedding space may differ from
/// SigLIP2 hidden dim; here we use same for simplicity).
const PROJ_DIM: usize = HIDDEN_DIM;

/// Build a SigLIP2-to-Granite projection kernel.
///
/// Input: `[NUM_PATCHES, HIDDEN_DIM]` (Variable, SigLIP2 encoder output).
/// Output: `[NUM_PATCHES, PROJ_DIM]` (projected to Granite embedding space).
///
/// Linear projection maps vision encoder output to LM embedding space.
/// In practice, this may include multiple linear layers or an MLP adapter.
fn build_siglip2_to_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_to_projection");

    let input = b.add_input("vision_output", &[NUM_PATCHES, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_weight", &[PROJ_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_bias", &[PROJ_DIM]);

    let out = b.add_linear(input, proj_w, Some(proj_b), &[NUM_PATCHES, PROJ_DIM]);

    b.build(out).expect("valid SigLIP2-to-projection kernel")
}

/// Bindings for SigLIP2-to-Granite projection.
fn siglip2_to_projection_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[PROJ_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[PROJ_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,               // vision_output
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
        TensorParamBinding::ConstantTensor(proj_b), // proj_bias
    ]
}

/// IBP bounds propagate through SigLIP2-to-Granite projection.
///
/// Linear projection from vision encoder output to LM embedding space.
/// Output bounds scale with weight * input range.
#[test]
fn test_siglip2_to_projection_ibp() {
    let def = build_siglip2_to_projection_kernel();
    let bindings = siglip2_to_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through SigLIP2-to-projection");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, PROJ_DIM],
        "projection output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 to Granite projection IBP (features [-2,2]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Linear with D=64, weight=0.02, input in [-2, 2]:
    // max output = sum(|w_i| * 2.0) = 64 * 0.02 * 2 = 2.56
    assert!(
        hi_max < 10.0,
        "projection upper should be < 10 with small weights, got {hi_max}"
    );
}

// ===========================================================================
// 13. Full VLM Compose IBP (end-to-end)
// ===========================================================================

/// Build a full VLM (Vision-Language Model) compose kernel.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image pixels in [0, 1]).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]` (Granite decoder FFN output).
///
/// End-to-end pipeline:
///   Patch embedding (Conv2d -> reshape -> transpose)
///   -> SigLIP2 encoder (LayerNorm -> attention -> residual -> LayerNorm -> MLP -> residual)
///   -> Vision projection (Linear)
///   -> Granite decoder FFN (RMSNorm -> SwiGLU)
fn build_full_vlm_compose_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("full_vlm_compose");

    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];
    let ffn_shape = [NUM_PATCHES, FFN_DIM];

    // --- Stage 1: Patch embedding ---
    let patch_w = b.add_input(
        "patch_proj_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_proj_bias", &[HIDDEN_DIM]);

    let conv_out = b.add_conv2d(
        image,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &patch_shape);

    // --- Stage 2: Simplified SigLIP2 encoder (single attention + MLP) ---
    let enc_ln1_eps = b.add_input("enc_ln1_eps", &[1]);
    let enc_ln1_w = b.add_input("enc_ln1_weight", &[HIDDEN_DIM]);
    let enc_ln1_b = b.add_input("enc_ln1_bias", &[HIDDEN_DIM]);
    let enc_normed1 = b.add_layer_norm(patches, enc_ln1_eps, 1, enc_ln1_w, enc_ln1_b, &patch_shape);

    let enc_q_w = b.add_input("enc_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_k_w = b.add_input("enc_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_v_w = b.add_input("enc_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_out_w = b.add_input("enc_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let enc_q = b.add_linear(enc_normed1, enc_q_w, None, &patch_shape);
    let enc_k = b.add_linear(enc_normed1, enc_k_w, None, &patch_shape);
    let enc_v = b.add_linear(enc_normed1, enc_v_w, None, &patch_shape);

    let scale = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();
    let enc_attn = b.add_attention(
        enc_q,
        enc_k,
        enc_v,
        AttentionMask::Standard,
        Some(scale),
        &patch_shape,
    );
    let enc_attn_out = b.add_linear(enc_attn, enc_out_w, None, &patch_shape);
    let enc_res1 = b.add_binary_add(patches, enc_attn_out, &patch_shape);

    let enc_ln2_eps = b.add_input("enc_ln2_eps", &[1]);
    let enc_ln2_w = b.add_input("enc_ln2_weight", &[HIDDEN_DIM]);
    let enc_ln2_b = b.add_input("enc_ln2_bias", &[HIDDEN_DIM]);
    let enc_normed2 =
        b.add_layer_norm(enc_res1, enc_ln2_eps, 1, enc_ln2_w, enc_ln2_b, &patch_shape);

    let enc_fc1_w = b.add_input("enc_fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let enc_fc2_w = b.add_input("enc_fc2_weight", &[HIDDEN_DIM, FFN_DIM]);

    let enc_h = b.add_linear(enc_normed2, enc_fc1_w, None, &ffn_shape);
    let enc_h = b.add_gelu(enc_h, &ffn_shape);
    let enc_ffn_out = b.add_linear(enc_h, enc_fc2_w, None, &patch_shape);
    let enc_out = b.add_binary_add(enc_res1, enc_ffn_out, &patch_shape);

    // --- Stage 3: Vision projection ---
    let vp_w = b.add_input("vis_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vp_b = b.add_input("vis_proj_bias", &[HIDDEN_DIM]);
    let projected = b.add_linear(enc_out, vp_w, Some(vp_b), &patch_shape);

    // --- Stage 4: Granite decoder FFN (RMSNorm -> SwiGLU) ---
    let dec_eps = b.add_input("dec_norm_eps", &[1]);
    let dec_norm_w = b.add_input("dec_norm_weight", &[HIDDEN_DIM]);
    let dec_normed = b.add_rms_norm(projected, dec_eps, 1, dec_norm_w, &patch_shape);

    let dec_gate_w = b.add_input("dec_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let dec_up_w = b.add_input("dec_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let dec_down_w = b.add_input("dec_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let dec_gate = b.add_linear(dec_normed, dec_gate_w, None, &ffn_shape);
    let dec_gate_sig = b.add_sigmoid(dec_gate, &ffn_shape);
    let dec_gate_act = b.add_binary_mul(dec_gate, dec_gate_sig, &ffn_shape);
    let dec_up = b.add_linear(dec_normed, dec_up_w, None, &ffn_shape);
    let dec_hidden = b.add_binary_mul(dec_gate_act, dec_up, &ffn_shape);
    let out = b.add_linear(dec_hidden, dec_down_w, None, &patch_shape);

    b.build(out).expect("valid full VLM compose kernel")
}

/// Bindings for full VLM compose.
fn full_vlm_compose_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                       // image
        TensorParamBinding::ConstantTensor(patch_w),        // patch_proj_weight
        TensorParamBinding::ConstantTensor(patch_b),        // patch_proj_bias
        TensorParamBinding::ConstantScalar(1e-5),           // enc_ln1_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()),   // enc_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()),   // enc_ln1_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_v_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // enc_ln2_eps
        TensorParamBinding::ConstantTensor(ln_w),           // enc_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b),           // enc_ln2_bias
        TensorParamBinding::ConstantTensor(fc1_w.clone()),  // enc_fc1_weight
        TensorParamBinding::ConstantTensor(fc2_w.clone()),  // enc_fc2_weight
        TensorParamBinding::ConstantTensor(attn_w),         // vis_proj_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)), // vis_proj_bias
        TensorParamBinding::ConstantScalar(1e-5), // dec_norm_eps
        TensorParamBinding::ConstantTensor(norm_w), // dec_norm_weight
        TensorParamBinding::ConstantTensor(fc1_w), // dec_gate_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // dec_up_weight
        TensorParamBinding::ConstantTensor(fc2_w), // dec_down_weight
    ]
}

/// IBP bounds propagate through full VLM pipeline.
///
/// End-to-end: Patch embed -> SigLIP2 encoder -> projection -> Granite FFN.
/// Verifies that bounds remain finite through the entire vision-language
/// model composition from image input to decoder FFN output.
#[test]
fn test_full_vlm_compose_ibp() {
    let def = build_full_vlm_compose_kernel();
    let bindings = full_vlm_compose_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full VLM compose");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "full VLM compose output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full VLM compose IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 14. Two-layer Granite decoder stack IBP/CROWN
// ===========================================================================

/// Build a 2-layer Granite decoder stack.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Two consecutive Granite decoder layers, each:
///   RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
/// Tests CROWN depth through repeated RMSNorm + SiLU + multiplicative gating.
fn build_granite_2layer_decoder_stack_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_2layer_decoder_stack");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    // ---- Layer 1 ----
    let l1_norm1_eps = b.add_input("l1_norm1_eps", &[1]);
    let l1_norm1_w = b.add_input("l1_norm1_weight", &[HIDDEN_DIM]);
    let l1_normed1 = b.add_rms_norm(input, l1_norm1_eps, 1, l1_norm1_w, &shape);

    let l1_q_w = b.add_input("l1_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l1_k_w = b.add_input("l1_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l1_v_w = b.add_input("l1_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l1_out_w = b.add_input("l1_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let l1_q = b.add_linear(l1_normed1, l1_q_w, None, &shape);
    let l1_k = b.add_linear(l1_normed1, l1_k_w, None, &shape);
    let l1_v = b.add_linear(l1_normed1, l1_v_w, None, &shape);
    let l1_attn = b.add_attention(l1_q, l1_k, l1_v, AttentionMask::Causal, Some(scale), &shape);
    let l1_attn_out = b.add_linear(l1_attn, l1_out_w, None, &shape);
    let l1_res1 = b.add_binary_add(input, l1_attn_out, &shape);

    let l1_norm2_eps = b.add_input("l1_norm2_eps", &[1]);
    let l1_norm2_w = b.add_input("l1_norm2_weight", &[HIDDEN_DIM]);
    let l1_normed2 = b.add_rms_norm(l1_res1, l1_norm2_eps, 1, l1_norm2_w, &shape);

    let l1_gate_w = b.add_input("l1_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l1_up_w = b.add_input("l1_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l1_down_w = b.add_input("l1_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let l1_gate = b.add_linear(l1_normed2, l1_gate_w, None, &ffn_shape);
    let l1_gate_sig = b.add_sigmoid(l1_gate, &ffn_shape);
    let l1_gate_act = b.add_binary_mul(l1_gate, l1_gate_sig, &ffn_shape);
    let l1_up = b.add_linear(l1_normed2, l1_up_w, None, &ffn_shape);
    let l1_hidden = b.add_binary_mul(l1_gate_act, l1_up, &ffn_shape);
    let l1_ffn_out = b.add_linear(l1_hidden, l1_down_w, None, &shape);
    let layer1_out = b.add_binary_add(l1_res1, l1_ffn_out, &shape);

    // ---- Layer 2 ----
    let l2_norm1_eps = b.add_input("l2_norm1_eps", &[1]);
    let l2_norm1_w = b.add_input("l2_norm1_weight", &[HIDDEN_DIM]);
    let l2_normed1 = b.add_rms_norm(layer1_out, l2_norm1_eps, 1, l2_norm1_w, &shape);

    let l2_q_w = b.add_input("l2_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l2_k_w = b.add_input("l2_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l2_v_w = b.add_input("l2_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let l2_out_w = b.add_input("l2_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let l2_q = b.add_linear(l2_normed1, l2_q_w, None, &shape);
    let l2_k = b.add_linear(l2_normed1, l2_k_w, None, &shape);
    let l2_v = b.add_linear(l2_normed1, l2_v_w, None, &shape);
    let l2_attn = b.add_attention(l2_q, l2_k, l2_v, AttentionMask::Causal, Some(scale), &shape);
    let l2_attn_out = b.add_linear(l2_attn, l2_out_w, None, &shape);
    let l2_res1 = b.add_binary_add(layer1_out, l2_attn_out, &shape);

    let l2_norm2_eps = b.add_input("l2_norm2_eps", &[1]);
    let l2_norm2_w = b.add_input("l2_norm2_weight", &[HIDDEN_DIM]);
    let l2_normed2 = b.add_rms_norm(l2_res1, l2_norm2_eps, 1, l2_norm2_w, &shape);

    let l2_gate_w = b.add_input("l2_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l2_up_w = b.add_input("l2_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l2_down_w = b.add_input("l2_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let l2_gate = b.add_linear(l2_normed2, l2_gate_w, None, &ffn_shape);
    let l2_gate_sig = b.add_sigmoid(l2_gate, &ffn_shape);
    let l2_gate_act = b.add_binary_mul(l2_gate, l2_gate_sig, &ffn_shape);
    let l2_up = b.add_linear(l2_normed2, l2_up_w, None, &ffn_shape);
    let l2_hidden = b.add_binary_mul(l2_gate_act, l2_up, &ffn_shape);
    let l2_ffn_out = b.add_linear(l2_hidden, l2_down_w, None, &shape);
    let out = b.add_binary_add(l2_res1, l2_ffn_out, &shape);

    b.build(out)
        .expect("valid Granite 2-layer decoder stack kernel")
}

/// Bindings for 2-layer Granite decoder stack.
fn granite_2layer_decoder_stack_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    // Layer 1
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // l1_norm1_eps
    bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // l1_norm1_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l1_q_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l1_k_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l1_v_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l1_out_weight
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // l1_norm2_eps
    bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // l1_norm2_weight
    bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // l1_gate_weight
    bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // l1_up_weight
    bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // l1_down_weight

    // Layer 2
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // l2_norm1_eps
    bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // l2_norm1_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l2_q_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l2_k_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // l2_v_weight
    bindings.push(TensorParamBinding::ConstantTensor(attn_w)); // l2_out_weight
    bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // l2_norm2_eps
    bindings.push(TensorParamBinding::ConstantTensor(norm_w)); // l2_norm2_weight
    bindings.push(TensorParamBinding::ConstantTensor(gate_w)); // l2_gate_weight
    bindings.push(TensorParamBinding::ConstantTensor(up_w)); // l2_up_weight
    bindings.push(TensorParamBinding::ConstantTensor(down_w)); // l2_down_weight

    bindings
}

/// IBP bounds propagate through 2-layer Granite decoder stack.
///
/// Two consecutive decoder layers with RMSNorm, causal attention, and SwiGLU.
/// Tests depth-2 bound propagation through the full Granite decoder architecture.
#[test]
fn test_granite_2layer_decoder_stack_ibp() {
    let def = build_granite_2layer_decoder_stack_kernel();
    let bindings = granite_2layer_decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-layer Granite decoder stack");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "2-layer decoder stack output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite 2-layer decoder stack IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds propagate through 2-layer Granite decoder stack.
///
/// Tests CROWN linearization depth through 2x RMSNorm, 2x softmax (attention),
/// 2x sigmoid (SiLU), and 2x multiplicative gating (McCormick envelope).
#[test]
fn test_granite_2layer_decoder_stack_crown() {
    let def = build_granite_2layer_decoder_stack_kernel();
    let bindings = granite_2layer_decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite 2-layer decoder stack: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 15. Three-layer Granite decoder stack (deeper composition)
// ===========================================================================

/// Build a 3-layer Granite decoder stack for deeper composition testing.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Three consecutive Granite decoder layers. Tests whether IBP bounds
/// remain finite through deeper transformer stacks, exercising the
/// bound-expansion behavior of repeated normalization + gating.
fn build_granite_3layer_decoder_stack_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_3layer_decoder_stack");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    // Helper closure emulated via inline expansion (3 layers)
    let mut prev = input;
    for layer_idx in 0..3 {
        let prefix = format!("l{layer_idx}");

        // Pre-attention RMSNorm
        let norm1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let norm1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(prev, norm1_eps, 1, norm1_w, &shape);

        // Self-attention
        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, q_w, None, &shape);
        let k = b.add_linear(normed1, k_w, None, &shape);
        let v = b.add_linear(normed1, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(prev, attn_out, &shape);

        // Pre-FFN RMSNorm
        let norm2_eps = b.add_input(&format!("{prefix}_norm2_eps"), &[1]);
        let norm2_w = b.add_input(&format!("{prefix}_norm2_weight"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

        // SwiGLU FFN
        let gate_w = b.add_input(&format!("{prefix}_gate_weight"), &[FFN_DIM, HIDDEN_DIM]);
        let up_w = b.add_input(&format!("{prefix}_up_weight"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("{prefix}_down_weight"), &[HIDDEN_DIM, FFN_DIM]);

        let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
        let gate_sig = b.add_sigmoid(gate, &ffn_shape);
        let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
        let up = b.add_linear(normed2, up_w, None, &ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
        let ffn_out = b.add_linear(hidden, down_w, None, &shape);
        prev = b.add_binary_add(res1, ffn_out, &shape);
    }

    b.build(prev)
        .expect("valid Granite 3-layer decoder stack kernel")
}

/// Bindings for 3-layer Granite decoder stack.
fn granite_3layer_decoder_stack_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _ in 0..3 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gate_weight
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // up_weight
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // down_weight
    }

    bindings
}

/// IBP bounds propagate through 3-layer Granite decoder stack.
///
/// Depth-3 composition: 3x (RMSNorm -> Attention -> RMSNorm -> SwiGLU + residuals).
/// Bounds should remain finite despite three layers of normalization and gating.
#[test]
fn test_granite_3layer_decoder_stack_ibp() {
    let def = build_granite_3layer_decoder_stack_kernel();
    let bindings = granite_3layer_decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3-layer Granite decoder stack");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "3-layer decoder stack output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite 3-layer decoder stack IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 16. Patch embedding -> 2 ViT blocks -> projection pipeline
// ===========================================================================

/// Build a patch embed -> 2 ViT blocks -> projection pipeline.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image pixels [0, 1]).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
///
/// Full SigLIP2 vision encoder pipeline: patch embedding (Conv2d -> reshape ->
/// transpose) followed by 2 stacked ViT blocks (LayerNorm -> Attention ->
/// residual -> LayerNorm -> GELU MLP -> residual) and a linear projection.
fn build_patch_embed_vit_stack_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("patch_embed_vit_stack_projection");

    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];
    let ffn_shape = [NUM_PATCHES, FFN_DIM];
    let scale = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();

    // --- Patch embedding ---
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        image,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let mut prev = b.add_transpose(reshaped, &[1, 0], &patch_shape);

    // --- 2 ViT blocks ---
    for layer_idx in 0..2 {
        let prefix = format!("vit{layer_idx}");

        let ln1_eps = b.add_input(&format!("{prefix}_ln1_eps"), &[1]);
        let ln1_w = b.add_input(&format!("{prefix}_ln1_weight"), &[HIDDEN_DIM]);
        let ln1_b = b.add_input(&format!("{prefix}_ln1_bias"), &[HIDDEN_DIM]);
        let normed1 = b.add_layer_norm(prev, ln1_eps, 1, ln1_w, ln1_b, &patch_shape);

        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, q_w, None, &patch_shape);
        let k = b.add_linear(normed1, k_w, None, &patch_shape);
        let v = b.add_linear(normed1, v_w, None, &patch_shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &patch_shape);
        let attn_out = b.add_linear(attn, out_w, None, &patch_shape);
        let res1 = b.add_binary_add(prev, attn_out, &patch_shape);

        let ln2_eps = b.add_input(&format!("{prefix}_ln2_eps"), &[1]);
        let ln2_w = b.add_input(&format!("{prefix}_ln2_weight"), &[HIDDEN_DIM]);
        let ln2_b = b.add_input(&format!("{prefix}_ln2_bias"), &[HIDDEN_DIM]);
        let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &patch_shape);

        let fc1_w = b.add_input(&format!("{prefix}_fc1_weight"), &[FFN_DIM, HIDDEN_DIM]);
        let fc2_w = b.add_input(&format!("{prefix}_fc2_weight"), &[HIDDEN_DIM, FFN_DIM]);

        let h = b.add_linear(normed2, fc1_w, None, &ffn_shape);
        let h = b.add_gelu(h, &ffn_shape);
        let ffn_out = b.add_linear(h, fc2_w, None, &patch_shape);
        prev = b.add_binary_add(res1, ffn_out, &patch_shape);
    }

    // --- Final projection ---
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_bias", &[HIDDEN_DIM]);
    let out = b.add_linear(prev, proj_w, Some(proj_b), &patch_shape);

    b.build(out)
        .expect("valid patch embed -> ViT stack -> projection kernel")
}

/// Bindings for patch embed -> ViT stack -> projection.
fn patch_embed_vit_stack_projection_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    let mut bindings = vec![
        TensorParamBinding::Variable,                // image
        TensorParamBinding::ConstantTensor(patch_w), // patch_weight
        TensorParamBinding::ConstantTensor(patch_b), // patch_bias
    ];

    // 2 ViT blocks
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln1_eps
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // ln1_weight
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // ln1_bias
        bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln2_eps
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // ln2_weight
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // ln2_bias
        bindings.push(TensorParamBinding::ConstantTensor(fc1_w.clone())); // fc1_weight
        bindings.push(TensorParamBinding::ConstantTensor(fc2_w.clone())); // fc2_weight
    }

    // Final projection
    bindings.push(TensorParamBinding::ConstantTensor(attn_w)); // proj_weight
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM]),
        0.0f32,
    ))); // proj_bias

    bindings
}

/// IBP through patch embed -> 2 ViT blocks -> projection.
///
/// Full SigLIP2 vision encoder pipeline from image pixels to projected features.
/// Tests multi-stage composition: Conv2d -> reshape -> transpose -> 2x encoder layers -> Linear.
#[test]
fn test_patch_embed_vit_stack_projection_ibp() {
    let def = build_patch_embed_vit_stack_projection_kernel();
    let bindings = patch_embed_vit_stack_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through patch embed -> ViT stack -> projection");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "patch embed -> ViT -> projection output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch embed -> 2 ViT -> projection IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 17. Token embedding -> decoder -> softmax pipeline
// ===========================================================================

/// Vocabulary size for token embedding tests.
const VOCAB_SIZE: usize = 32;

/// Build a token embedding -> decoder -> softmax pipeline.
///
/// Input: `[SEQ_LEN]` (Variable, integer token indices).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (softmax probability distribution).
///
/// Token embedding lookup -> single Granite decoder layer
/// (RMSNorm -> Attention -> SwiGLU) -> Linear LM head -> Softmax.
fn build_token_embed_decoder_softmax_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("token_embed_decoder_softmax");

    let input = b.add_input("token_ids", &[SEQ_LEN]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    // --- Token embedding ---
    let embed_w = b.add_input("embed_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let embedded = b.add_embedding(input, embed_w, &shape);

    // --- Single decoder layer ---
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(embedded, norm1_eps, 1, norm1_w, &shape);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(embedded, attn_out, &shape);

    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);
    let dec_out = b.add_binary_add(res1, ffn_out, &shape);

    // --- LM head + softmax ---
    let lm_head_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(dec_out, lm_head_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid token embed -> decoder -> softmax kernel")
}

/// Bindings for token embed -> decoder -> softmax.
fn token_embed_decoder_softmax_bindings() -> Vec<TensorParamBinding> {
    let embed_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let lm_head_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // token_ids
        TensorParamBinding::ConstantTensor(embed_w),        // embed_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(attn_w),         // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
        TensorParamBinding::ConstantTensor(lm_head_w),      // lm_head_weight
    ]
}

/// IBP through token embed -> decoder -> softmax pipeline.
///
/// End-to-end text generation pipeline: embedding lookup -> decoder -> LM head -> softmax.
/// Softmax output should be bounded in [0, 1] for each token position.
#[test]
fn test_token_embed_decoder_softmax_ibp() {
    let def = build_token_embed_decoder_softmax_kernel();
    let bindings = token_embed_decoder_softmax_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Token indices bounded in [0, VOCAB_SIZE-1]
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), (VOCAB_SIZE - 1) as f32),
    )
    .expect("valid token bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through token embed -> decoder -> softmax");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "token pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Token embed -> decoder -> softmax IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-6,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "softmax upper bound should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 18. Decoder with causal mask attention
// ===========================================================================

/// Build a Granite decoder block with explicit causal masking.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Uses `AttentionMask::Causal` to verify that causal attention preserves
/// bounded output. Position j attends only to positions <= j.
fn build_granite_causal_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_causal_decoder");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    // RMSNorm -> Causal Attention -> residual
    let eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);

    // Causal attention: each position attends only to past + self
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(input, attn_out, &shape);

    b.build(out).expect("valid Granite causal decoder kernel")
}

/// Bindings for Granite causal decoder.
fn granite_causal_decoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-5),           // norm_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(attn_w),         // out_weight
    ]
}

/// IBP through Granite decoder with causal mask attention.
///
/// Verifies that causal masking (position j attends only to <= j)
/// preserves bounded output through the attention mechanism.
#[test]
fn test_granite_causal_decoder_ibp() {
    let def = build_granite_causal_decoder_kernel();
    let bindings = granite_causal_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Granite causal decoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "causal decoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite causal decoder IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through Granite decoder with causal mask attention.
///
/// Tests CROWN linearization through RMSNorm + softmax (causal attention).
#[test]
fn test_granite_causal_decoder_crown() {
    let def = build_granite_causal_decoder_kernel();
    let bindings = granite_causal_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite causal decoder: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 19. Cross-attention: vision features as K/V, text as query
// ===========================================================================

/// Build a cross-attention block: text queries attend to vision features.
///
/// Text input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Vision features: `[NUM_PATCHES, HIDDEN_DIM]` (Constant, from frozen encoder).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Q is projected from text, K and V from vision. This is the core
/// cross-modal fusion mechanism in vision-language models.
fn build_cross_attention_vision_text_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_attention_vision_text");

    let text = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let vision = b.add_input("vision_features", &[NUM_PATCHES, HIDDEN_DIM]);
    let text_shape = [SEQ_LEN, HIDDEN_DIM];
    let vision_shape = [NUM_PATCHES, HIDDEN_DIM];
    let scale = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();

    // Q from text, K/V from vision
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(text, q_w, None, &text_shape);
    let k = b.add_linear(vision, k_w, None, &vision_shape);
    let v = b.add_linear(vision, v_w, None, &vision_shape);

    // Cross-attention: text queries, vision keys/values
    // Output shape follows Q: [SEQ_LEN, HIDDEN_DIM]
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &text_shape);
    let attn_out = b.add_linear(attn, out_w, None, &text_shape);

    // Residual connection with text input
    let out = b.add_binary_add(text, attn_out, &text_shape);

    b.build(out)
        .expect("valid cross-attention vision-text kernel")
}

/// Bindings for cross-attention vision-text.
fn cross_attention_vision_text_bindings() -> Vec<TensorParamBinding> {
    let vision_features = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), 0.5f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable, // text_hidden (Variable)
        TensorParamBinding::ConstantTensor(vision_features), // vision_features (frozen)
        TensorParamBinding::ConstantTensor(attn_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(attn_w), // out_weight
    ]
}

/// IBP through cross-attention: text queries attend to vision features.
///
/// Verifies bounds propagation for cross-modal fusion: Q from text (Variable),
/// K/V from vision (Constant). Output shape follows Q: [SEQ_LEN, HIDDEN_DIM].
#[test]
fn test_cross_attention_vision_text_ibp() {
    let def = build_cross_attention_vision_text_kernel();
    let bindings = cross_attention_vision_text_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attention vision-text");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "cross-attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attention vision-text IBP (text [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through cross-attention: text queries attend to vision features.
///
/// Tests CROWN linearization through softmax in cross-attention where
/// K/V are constant (from frozen vision encoder) and Q is variable.
#[test]
fn test_cross_attention_vision_text_crown() {
    let def = build_cross_attention_vision_text_kernel();
    let bindings = cross_attention_vision_text_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attention vision-text: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 20. Vision encoder -> projection -> decoder cross-attention
// ===========================================================================

/// Build a vision encoder -> projection -> decoder cross-attention pipeline.
///
/// Image input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[SEQ_LEN, HIDDEN_DIM]` (decoder output after cross-attention).
///
/// Pipeline: Patch embed -> SigLIP2 encoder (1 block) -> Linear projection
/// -> Cross-attention (text Q, vision K/V) with text-side residual.
fn build_vision_enc_proj_decoder_xattn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vision_enc_proj_decoder_xattn");

    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let text = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];
    let text_shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [NUM_PATCHES, FFN_DIM];
    let scale_enc = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();
    let scale_dec = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    // --- Patch embedding ---
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        image,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &patch_shape);

    // --- Single SigLIP2 encoder block ---
    let ln1_eps = b.add_input("enc_ln1_eps", &[1]);
    let ln1_w = b.add_input("enc_ln1_weight", &[HIDDEN_DIM]);
    let ln1_b = b.add_input("enc_ln1_bias", &[HIDDEN_DIM]);
    let normed1 = b.add_layer_norm(patches, ln1_eps, 1, ln1_w, ln1_b, &patch_shape);

    let enc_q_w = b.add_input("enc_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_k_w = b.add_input("enc_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_v_w = b.add_input("enc_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_out_w = b.add_input("enc_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let enc_q = b.add_linear(normed1, enc_q_w, None, &patch_shape);
    let enc_k = b.add_linear(normed1, enc_k_w, None, &patch_shape);
    let enc_v = b.add_linear(normed1, enc_v_w, None, &patch_shape);
    let enc_attn = b.add_attention(
        enc_q,
        enc_k,
        enc_v,
        AttentionMask::Standard,
        Some(scale_enc),
        &patch_shape,
    );
    let enc_attn_out = b.add_linear(enc_attn, enc_out_w, None, &patch_shape);
    let enc_res1 = b.add_binary_add(patches, enc_attn_out, &patch_shape);

    let ln2_eps = b.add_input("enc_ln2_eps", &[1]);
    let ln2_w = b.add_input("enc_ln2_weight", &[HIDDEN_DIM]);
    let ln2_b = b.add_input("enc_ln2_bias", &[HIDDEN_DIM]);
    let normed2 = b.add_layer_norm(enc_res1, ln2_eps, 1, ln2_w, ln2_b, &patch_shape);

    let fc1_w = b.add_input("enc_fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc2_w = b.add_input("enc_fc2_weight", &[HIDDEN_DIM, FFN_DIM]);

    let h = b.add_linear(normed2, fc1_w, None, &ffn_shape);
    let h = b.add_gelu(h, &ffn_shape);
    let ffn_out = b.add_linear(h, fc2_w, None, &patch_shape);
    let enc_out = b.add_binary_add(enc_res1, ffn_out, &patch_shape);

    // --- Vision projection ---
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_bias", &[HIDDEN_DIM]);
    let vision_projected = b.add_linear(enc_out, proj_w, Some(proj_b), &patch_shape);

    // --- Decoder cross-attention: text Q, vision K/V ---
    let xq_w = b.add_input("xattn_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let xk_w = b.add_input("xattn_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let xv_w = b.add_input("xattn_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let xout_w = b.add_input("xattn_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let xq = b.add_linear(text, xq_w, None, &text_shape);
    let xk = b.add_linear(vision_projected, xk_w, None, &patch_shape);
    let xv = b.add_linear(vision_projected, xv_w, None, &patch_shape);
    let xattn = b.add_attention(
        xq,
        xk,
        xv,
        AttentionMask::Standard,
        Some(scale_dec),
        &text_shape,
    );
    let xattn_out = b.add_linear(xattn, xout_w, None, &text_shape);
    let out = b.add_binary_add(text, xattn_out, &text_shape);

    b.build(out)
        .expect("valid vision enc -> proj -> decoder xattn kernel")
}

/// Bindings for vision enc -> projection -> decoder cross-attention.
fn vision_enc_proj_decoder_xattn_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let text_hidden = ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.1f32);

    vec![
        TensorParamBinding::Variable,                       // image (Variable)
        TensorParamBinding::ConstantTensor(text_hidden),    // text_hidden (frozen)
        TensorParamBinding::ConstantTensor(patch_w),        // patch_weight
        TensorParamBinding::ConstantTensor(patch_b),        // patch_bias
        TensorParamBinding::ConstantScalar(1e-5),           // enc_ln1_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()),   // enc_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()),   // enc_ln1_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_v_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // enc_ln2_eps
        TensorParamBinding::ConstantTensor(ln_w),           // enc_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b),           // enc_ln2_bias
        TensorParamBinding::ConstantTensor(fc1_w),          // enc_fc1_weight
        TensorParamBinding::ConstantTensor(fc2_w),          // enc_fc2_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // proj_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)), // proj_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // xattn_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // xattn_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // xattn_v_weight
        TensorParamBinding::ConstantTensor(attn_w),         // xattn_out_weight
    ]
}

/// IBP through vision encoder -> projection -> decoder cross-attention.
///
/// Full cross-modal pipeline: image -> patch embed -> encoder -> projection
/// -> cross-attention with frozen text queries. Tests depth through multiple
/// non-linearities: LayerNorm, GELU, softmax (encoder + decoder attention).
#[test]
fn test_vision_enc_proj_decoder_xattn_ibp() {
    let def = build_vision_enc_proj_decoder_xattn_kernel();
    let bindings = vision_enc_proj_decoder_xattn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through vision enc -> proj -> decoder xattn");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "vision enc -> proj -> xattn output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vision enc -> proj -> decoder xattn IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 21. End-to-end classification: vision + text -> logits -> softmax
// ===========================================================================

/// Number of classes for classification.
const NUM_CLASSES: usize = 8;

/// Build an end-to-end VLM classification pipeline.
///
/// Image input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[NUM_CLASSES]` (softmax class probabilities).
///
/// Pipeline: Patch embed -> SigLIP2 encoder (1 block) -> mean pool over
/// patches -> Linear classifier -> Softmax. Tests full image-to-probability
/// pipeline including spatial aggregation (mean reduction).
fn build_vlm_classification_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vlm_classification");

    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];
    let ffn_shape = [NUM_PATCHES, FFN_DIM];
    let scale = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();

    // --- Patch embedding ---
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        image,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &patch_shape);

    // --- Single SigLIP2 encoder block ---
    let ln1_eps = b.add_input("ln1_eps", &[1]);
    let ln1_w = b.add_input("ln1_weight", &[HIDDEN_DIM]);
    let ln1_b = b.add_input("ln1_bias", &[HIDDEN_DIM]);
    let normed1 = b.add_layer_norm(patches, ln1_eps, 1, ln1_w, ln1_b, &patch_shape);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &patch_shape);
    let k = b.add_linear(normed1, k_w, None, &patch_shape);
    let v = b.add_linear(normed1, v_w, None, &patch_shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &patch_shape);
    let attn_out = b.add_linear(attn, out_w, None, &patch_shape);
    let res1 = b.add_binary_add(patches, attn_out, &patch_shape);

    let ln2_eps = b.add_input("ln2_eps", &[1]);
    let ln2_w = b.add_input("ln2_weight", &[HIDDEN_DIM]);
    let ln2_b = b.add_input("ln2_bias", &[HIDDEN_DIM]);
    let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &patch_shape);

    let fc1_w = b.add_input("fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc2_w = b.add_input("fc2_weight", &[HIDDEN_DIM, FFN_DIM]);

    let h = b.add_linear(normed2, fc1_w, None, &ffn_shape);
    let h = b.add_gelu(h, &ffn_shape);
    let ffn_out = b.add_linear(h, fc2_w, None, &patch_shape);
    let enc_out = b.add_binary_add(res1, ffn_out, &patch_shape);

    // --- Mean pool over patches: [NUM_PATCHES, HIDDEN_DIM] -> [HIDDEN_DIM] ---
    let pooled = b.add_reduce(enc_out, ReduceOp::Mean, 0, false, &[HIDDEN_DIM]);

    // --- Classification head: Linear -> Softmax ---
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(pooled, cls_w, Some(cls_b), &[NUM_CLASSES]);
    let out = b.add_softmax(logits, 0, &[NUM_CLASSES]);

    b.build(out).expect("valid VLM classification kernel")
}

/// Bindings for VLM classification pipeline.
fn vlm_classification_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let cls_w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, HIDDEN_DIM]), WEIGHT_MAG);
    let cls_b = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                       // image
        TensorParamBinding::ConstantTensor(patch_w),        // patch_weight
        TensorParamBinding::ConstantTensor(patch_b),        // patch_bias
        TensorParamBinding::ConstantScalar(1e-5),           // ln1_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()),   // ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()),   // ln1_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(attn_w),         // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // ln2_eps
        TensorParamBinding::ConstantTensor(ln_w),           // ln2_weight
        TensorParamBinding::ConstantTensor(ln_b),           // ln2_bias
        TensorParamBinding::ConstantTensor(fc1_w),          // fc1_weight
        TensorParamBinding::ConstantTensor(fc2_w),          // fc2_weight
        TensorParamBinding::ConstantTensor(cls_w),          // cls_weight
        TensorParamBinding::ConstantTensor(cls_b),          // cls_bias
    ]
}

/// IBP through end-to-end VLM classification.
///
/// Image -> patch embed -> encoder -> mean pool -> Linear -> Softmax.
/// Verifies that softmax output probabilities are in [0, 1].
#[test]
fn test_vlm_classification_ibp() {
    let def = build_vlm_classification_kernel();
    let bindings = vlm_classification_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through VLM classification");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_CLASSES],
        "classification output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("VLM classification IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-6,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "softmax upper bound should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 22. Full VLM: patch embed -> vision encoder -> projection -> decoder -> LM head
// ===========================================================================

/// Build a full VLM pipeline with decoder and LM head.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image pixels [0, 1]).
/// Output: `[NUM_PATCHES, VOCAB_SIZE]` (softmax token probabilities).
///
/// End-to-end: Patch embed -> SigLIP2 encoder -> vision projection ->
/// Granite decoder (RMSNorm -> attention -> SwiGLU) -> LM head -> Softmax.
/// This is the most complete VLM pipeline test.
fn build_full_vlm_decoder_lm_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("full_vlm_decoder_lm_head");

    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];
    let enc_ffn_shape = [NUM_PATCHES, FFN_DIM];
    let dec_ffn_shape = [NUM_PATCHES, FFN_DIM];
    let scale_enc = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();
    let scale_dec = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    // --- Patch embedding ---
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        image,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &patch_shape);

    // --- SigLIP2 encoder (1 block) ---
    let enc_ln1_eps = b.add_input("enc_ln1_eps", &[1]);
    let enc_ln1_w = b.add_input("enc_ln1_weight", &[HIDDEN_DIM]);
    let enc_ln1_b = b.add_input("enc_ln1_bias", &[HIDDEN_DIM]);
    let enc_normed1 = b.add_layer_norm(patches, enc_ln1_eps, 1, enc_ln1_w, enc_ln1_b, &patch_shape);

    let enc_q_w = b.add_input("enc_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_k_w = b.add_input("enc_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_v_w = b.add_input("enc_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_out_w = b.add_input("enc_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let enc_q = b.add_linear(enc_normed1, enc_q_w, None, &patch_shape);
    let enc_k = b.add_linear(enc_normed1, enc_k_w, None, &patch_shape);
    let enc_v = b.add_linear(enc_normed1, enc_v_w, None, &patch_shape);
    let enc_attn = b.add_attention(
        enc_q,
        enc_k,
        enc_v,
        AttentionMask::Standard,
        Some(scale_enc),
        &patch_shape,
    );
    let enc_attn_out = b.add_linear(enc_attn, enc_out_w, None, &patch_shape);
    let enc_res1 = b.add_binary_add(patches, enc_attn_out, &patch_shape);

    let enc_ln2_eps = b.add_input("enc_ln2_eps", &[1]);
    let enc_ln2_w = b.add_input("enc_ln2_weight", &[HIDDEN_DIM]);
    let enc_ln2_b = b.add_input("enc_ln2_bias", &[HIDDEN_DIM]);
    let enc_normed2 =
        b.add_layer_norm(enc_res1, enc_ln2_eps, 1, enc_ln2_w, enc_ln2_b, &patch_shape);

    let enc_fc1_w = b.add_input("enc_fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let enc_fc2_w = b.add_input("enc_fc2_weight", &[HIDDEN_DIM, FFN_DIM]);

    let enc_h = b.add_linear(enc_normed2, enc_fc1_w, None, &enc_ffn_shape);
    let enc_h = b.add_gelu(enc_h, &enc_ffn_shape);
    let enc_ffn_out = b.add_linear(enc_h, enc_fc2_w, None, &patch_shape);
    let enc_out = b.add_binary_add(enc_res1, enc_ffn_out, &patch_shape);

    // --- Vision projection ---
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_bias", &[HIDDEN_DIM]);
    let projected = b.add_linear(enc_out, proj_w, Some(proj_b), &patch_shape);

    // --- Granite decoder (1 layer) ---
    let dec_norm1_eps = b.add_input("dec_norm1_eps", &[1]);
    let dec_norm1_w = b.add_input("dec_norm1_weight", &[HIDDEN_DIM]);
    let dec_normed1 = b.add_rms_norm(projected, dec_norm1_eps, 1, dec_norm1_w, &patch_shape);

    let dec_q_w = b.add_input("dec_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let dec_k_w = b.add_input("dec_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let dec_v_w = b.add_input("dec_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let dec_out_w = b.add_input("dec_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let dec_q = b.add_linear(dec_normed1, dec_q_w, None, &patch_shape);
    let dec_k = b.add_linear(dec_normed1, dec_k_w, None, &patch_shape);
    let dec_v = b.add_linear(dec_normed1, dec_v_w, None, &patch_shape);
    let dec_attn = b.add_attention(
        dec_q,
        dec_k,
        dec_v,
        AttentionMask::Causal,
        Some(scale_dec),
        &patch_shape,
    );
    let dec_attn_out = b.add_linear(dec_attn, dec_out_w, None, &patch_shape);
    let dec_res1 = b.add_binary_add(projected, dec_attn_out, &patch_shape);

    let dec_norm2_eps = b.add_input("dec_norm2_eps", &[1]);
    let dec_norm2_w = b.add_input("dec_norm2_weight", &[HIDDEN_DIM]);
    let dec_normed2 = b.add_rms_norm(dec_res1, dec_norm2_eps, 1, dec_norm2_w, &patch_shape);

    let dec_gate_w = b.add_input("dec_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let dec_up_w = b.add_input("dec_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let dec_down_w = b.add_input("dec_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let dec_gate = b.add_linear(dec_normed2, dec_gate_w, None, &dec_ffn_shape);
    let dec_gate_sig = b.add_sigmoid(dec_gate, &dec_ffn_shape);
    let dec_gate_act = b.add_binary_mul(dec_gate, dec_gate_sig, &dec_ffn_shape);
    let dec_up = b.add_linear(dec_normed2, dec_up_w, None, &dec_ffn_shape);
    let dec_hidden = b.add_binary_mul(dec_gate_act, dec_up, &dec_ffn_shape);
    let dec_ffn_out = b.add_linear(dec_hidden, dec_down_w, None, &patch_shape);
    let dec_out = b.add_binary_add(dec_res1, dec_ffn_out, &patch_shape);

    // --- LM head + softmax ---
    let lm_head_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(dec_out, lm_head_w, None, &[NUM_PATCHES, VOCAB_SIZE]);
    let out = b.add_softmax(logits, 1, &[NUM_PATCHES, VOCAB_SIZE]);

    b.build(out)
        .expect("valid full VLM decoder + LM head kernel")
}

/// Bindings for full VLM decoder + LM head.
fn full_vlm_decoder_lm_head_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let lm_head_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                // image
        TensorParamBinding::ConstantTensor(patch_w), // patch_weight
        TensorParamBinding::ConstantTensor(patch_b), // patch_bias
        // Encoder
        TensorParamBinding::ConstantScalar(1e-5), // enc_ln1_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // enc_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // enc_ln1_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_v_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc_out_weight
        TensorParamBinding::ConstantScalar(1e-5), // enc_ln2_eps
        TensorParamBinding::ConstantTensor(ln_w), // enc_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b), // enc_ln2_bias
        TensorParamBinding::ConstantTensor(fc1_w.clone()), // enc_fc1_weight
        TensorParamBinding::ConstantTensor(fc2_w.clone()), // enc_fc2_weight
        // Projection
        TensorParamBinding::ConstantTensor(attn_w.clone()), // proj_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)), // proj_bias
        // Decoder
        TensorParamBinding::ConstantScalar(1e-5), // dec_norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // dec_norm1_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // dec_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // dec_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // dec_v_weight
        TensorParamBinding::ConstantTensor(attn_w), // dec_out_weight
        TensorParamBinding::ConstantScalar(1e-5), // dec_norm2_eps
        TensorParamBinding::ConstantTensor(norm_w), // dec_norm2_weight
        TensorParamBinding::ConstantTensor(fc1_w), // dec_gate_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // dec_up_weight
        TensorParamBinding::ConstantTensor(fc2_w), // dec_down_weight
        // LM head
        TensorParamBinding::ConstantTensor(lm_head_w), // lm_head_weight
    ]
}

/// IBP through full VLM: patch embed -> encoder -> projection -> decoder -> LM head.
///
/// Most complete VLM pipeline test. Verifies bounds through all stages:
/// Conv2d -> LayerNorm -> attention -> GELU MLP -> Linear projection ->
/// RMSNorm -> causal attention -> SiLU gating -> Linear -> Softmax.
/// Softmax output must be in [0, 1].
#[test]
fn test_full_vlm_decoder_lm_head_ibp() {
    let def = build_full_vlm_decoder_lm_head_kernel();
    let bindings = full_vlm_decoder_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full VLM decoder + LM head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, VOCAB_SIZE],
        "full VLM + LM head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full VLM decoder + LM head IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-6,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "softmax upper bound should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 23. Full VLM compose CROWN propagation
// ===========================================================================

/// CROWN bounds through full VLM compose pipeline.
///
/// End-to-end: Patch embed -> SigLIP2 encoder -> projection -> Granite FFN.
/// Tests CROWN linearization through the deepest pipeline: Conv2d -> LayerNorm
/// -> softmax (attention) -> GELU -> RMSNorm -> sigmoid (SiLU) -> multiplicative
/// gating. Uses IbpValidated mode per nn engineering rules.
#[test]
fn test_full_vlm_compose_crown() {
    let def = build_full_vlm_compose_kernel();
    let bindings = full_vlm_compose_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full VLM compose: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 24. SigLIP2 4-Block Encoder Stack
// ===========================================================================

/// Additional dimensions for deep-stack and multi-resolution tests.
const IMG_SIZE_64: usize = 64;
const GRID_SIZE_64: usize = IMG_SIZE_64 / PATCH_SIZE; // 4
const NUM_PATCHES_64: usize = GRID_SIZE_64 * GRID_SIZE_64; // 16
const PATCH_SIZE_8: usize = 8;
const IMG_SIZE_48: usize = 48;
const GRID_SIZE_48_P8: usize = IMG_SIZE_48 / PATCH_SIZE_8; // 6
const NUM_PATCHES_48_P8: usize = GRID_SIZE_48_P8 * GRID_SIZE_48_P8; // 36

/// Build a 4-block SigLIP2 encoder stack using a loop-based builder.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Each block: LayerNorm -> MHA -> residual -> LayerNorm -> GELU MLP -> residual.
/// Tests CROWN depth through 4 consecutive encoder layers.
fn build_siglip2_4block_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_4block_encoder");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer in 0..4 {
        let prefix = format!("l{layer}");

        // Pre-attention LayerNorm
        let ln1_eps = b.add_input(&format!("{prefix}_ln1_eps"), &[1]);
        let ln1_w = b.add_input(&format!("{prefix}_ln1_weight"), &[HIDDEN_DIM]);
        let ln1_b = b.add_input(&format!("{prefix}_ln1_bias"), &[HIDDEN_DIM]);
        let normed = b.add_layer_norm(current, ln1_eps, 1, ln1_w, ln1_b, &shape);

        // Multi-head attention
        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed, q_w, None, &shape);
        let k = b.add_linear(normed, k_w, None, &shape);
        let v = b.add_linear(normed, v_w, None, &shape);

        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(current, attn_out, &shape);

        // Pre-FFN LayerNorm
        let ln2_eps = b.add_input(&format!("{prefix}_ln2_eps"), &[1]);
        let ln2_w = b.add_input(&format!("{prefix}_ln2_weight"), &[HIDDEN_DIM]);
        let ln2_b = b.add_input(&format!("{prefix}_ln2_bias"), &[HIDDEN_DIM]);
        let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &shape);

        // GELU MLP
        let fc1_w = b.add_input(&format!("{prefix}_fc1_weight"), &[FFN_DIM, HIDDEN_DIM]);
        let fc2_w = b.add_input(&format!("{prefix}_fc2_weight"), &[HIDDEN_DIM, FFN_DIM]);

        let h = b.add_linear(normed2, fc1_w, None, &ffn_shape);
        let h = b.add_gelu(h, &ffn_shape);
        let ffn_out = b.add_linear(h, fc2_w, None, &shape);
        current = b.add_binary_add(res1, ffn_out, &shape);
    }

    b.build(current)
        .expect("valid SigLIP2 4-block encoder kernel")
}

/// Bindings for 4-block SigLIP2 encoder stack.
fn siglip2_4block_encoder_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
    for _layer in 0..4 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln1_eps
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // ln1_weight
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // ln1_bias
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln2_eps
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // ln2_weight
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // ln2_bias
        bindings.push(TensorParamBinding::ConstantTensor(fc1_w.clone())); // fc1_weight
        bindings.push(TensorParamBinding::ConstantTensor(fc2_w.clone())); // fc2_weight
    }
    bindings
}

/// IBP bounds through 4-block SigLIP2 encoder stack.
///
/// Tests bound propagation depth through 4 consecutive encoder layers
/// (4x LayerNorm + 4x Attention + 4x GELU MLP + 8x residual).
#[test]
fn test_siglip2_4block_encoder_ibp() {
    let def = build_siglip2_4block_encoder_kernel();
    let bindings = siglip2_4block_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-block SigLIP2 encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "4-block encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 4-block encoder IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through 4-block SigLIP2 encoder stack.
///
/// Tests CROWN linearization depth through 4 consecutive encoder layers.
/// Uses IbpValidated mode per nn engineering rules.
#[test]
fn test_siglip2_4block_encoder_crown() {
    let def = build_siglip2_4block_encoder_kernel();
    let bindings = siglip2_4block_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 4-block encoder: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 25. SigLIP2 8-Block Encoder Stack (Deepest)
// ===========================================================================

/// Build an 8-block SigLIP2 encoder stack -- deepest encoder test.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// This is the deepest encoder stack test, exercising bound propagation
/// through 8 LayerNorm + 8 Attention + 8 GELU MLP + 16 residual connections.
fn build_siglip2_8block_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_8block_encoder");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer in 0..8 {
        let prefix = format!("l{layer}");

        let ln1_eps = b.add_input(&format!("{prefix}_ln1_eps"), &[1]);
        let ln1_w = b.add_input(&format!("{prefix}_ln1_weight"), &[HIDDEN_DIM]);
        let ln1_b = b.add_input(&format!("{prefix}_ln1_bias"), &[HIDDEN_DIM]);
        let normed = b.add_layer_norm(current, ln1_eps, 1, ln1_w, ln1_b, &shape);

        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed, q_w, None, &shape);
        let k = b.add_linear(normed, k_w, None, &shape);
        let v = b.add_linear(normed, v_w, None, &shape);

        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(current, attn_out, &shape);

        let ln2_eps = b.add_input(&format!("{prefix}_ln2_eps"), &[1]);
        let ln2_w = b.add_input(&format!("{prefix}_ln2_weight"), &[HIDDEN_DIM]);
        let ln2_b = b.add_input(&format!("{prefix}_ln2_bias"), &[HIDDEN_DIM]);
        let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &shape);

        let fc1_w = b.add_input(&format!("{prefix}_fc1_weight"), &[FFN_DIM, HIDDEN_DIM]);
        let fc2_w = b.add_input(&format!("{prefix}_fc2_weight"), &[HIDDEN_DIM, FFN_DIM]);

        let h = b.add_linear(normed2, fc1_w, None, &ffn_shape);
        let h = b.add_gelu(h, &ffn_shape);
        let ffn_out = b.add_linear(h, fc2_w, None, &shape);
        current = b.add_binary_add(res1, ffn_out, &shape);
    }

    b.build(current)
        .expect("valid SigLIP2 8-block encoder kernel")
}

/// Bindings for 8-block SigLIP2 encoder stack.
fn siglip2_8block_encoder_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
    for _layer in 0..8 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(fc1_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(fc2_w.clone()));
    }
    bindings
}

/// IBP bounds through 8-block SigLIP2 encoder (deepest encoder test).
///
/// Exercises bound propagation through 8 encoder layers -- twice the
/// existing 2-layer stack tests. Tests whether IBP bounds remain finite
/// through the deepest encoder stack.
#[test]
fn test_siglip2_8block_encoder_ibp() {
    let def = build_siglip2_8block_encoder_kernel();
    let bindings = siglip2_8block_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 8-block SigLIP2 encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "8-block encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 8-block encoder IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 26. Vision Projection MLP Adapter
// ===========================================================================

/// Build a vision projection MLP adapter (LayerNorm -> Linear -> GELU -> Linear).
///
/// Input: `[NUM_PATCHES, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
///
/// Multi-layer adapter mapping vision features to LM embedding space,
/// more representative than a single linear projection.
fn build_vision_projection_mlp_adapter_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vision_projection_mlp_adapter");

    let input = b.add_input("vision_features", &[NUM_PATCHES, HIDDEN_DIM]);
    let shape = [NUM_PATCHES, HIDDEN_DIM];
    let ffn_shape = [NUM_PATCHES, FFN_DIM];

    // LayerNorm before projection
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &shape);

    // MLP: Linear -> GELU -> Linear
    let fc1_w = b.add_input("fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc2_w = b.add_input("fc2_weight", &[HIDDEN_DIM, FFN_DIM]);

    let h = b.add_linear(normed, fc1_w, None, &ffn_shape);
    let h = b.add_gelu(h, &ffn_shape);
    let out = b.add_linear(h, fc2_w, None, &shape);

    b.build(out)
        .expect("valid vision projection MLP adapter kernel")
}

/// Bindings for vision projection MLP adapter.
fn vision_projection_mlp_adapter_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,              // vision_features
        TensorParamBinding::ConstantScalar(1e-5),  // ln_eps
        TensorParamBinding::ConstantTensor(ln_w),  // ln_weight
        TensorParamBinding::ConstantTensor(ln_b),  // ln_bias
        TensorParamBinding::ConstantTensor(fc1_w), // fc1_weight
        TensorParamBinding::ConstantTensor(fc2_w), // fc2_weight
    ]
}

/// IBP bounds through vision projection MLP adapter.
///
/// LayerNorm -> Linear -> GELU -> Linear with image-domain input.
#[test]
fn test_vision_projection_mlp_adapter_ibp() {
    let def = build_vision_projection_mlp_adapter_kernel();
    let bindings = vision_projection_mlp_adapter_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through vision projection MLP adapter");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "vision projection MLP adapter output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vision projection MLP adapter IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through vision projection MLP adapter.
///
/// Tests CROWN linearization through LayerNorm + GELU in the projection
/// adapter. Uses IbpValidated mode.
#[test]
fn test_vision_projection_mlp_adapter_crown() {
    let def = build_vision_projection_mlp_adapter_kernel();
    let bindings = vision_projection_mlp_adapter_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vision projection MLP adapter: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 27. Cross-Modal Attention (text Q, vision K/V)
// ===========================================================================

/// Build cross-modal attention where queries come from text, K/V from vision.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- text hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Vision K/V are constant tensors (frozen encoder output).
/// RMSNorm before attention, residual after.
fn build_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_attention_rmsnorm");

    let text_input = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let _cross_kv_shape = [NUM_PATCHES, HIDDEN_DIM];

    // Pre-attention RMSNorm on text
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed_text = b.add_rms_norm(text_input, norm_eps, 1, norm_w, &shape);

    // Text Q projection
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let q = b.add_linear(normed_text, q_w, None, &shape);

    // Vision K/V (constant -- frozen encoder output, pre-projected)
    let vision_k = b.add_input("vision_k", &[NUM_PATCHES, HIDDEN_DIM]);
    let vision_v = b.add_input("vision_v", &[NUM_PATCHES, HIDDEN_DIM]);

    // Cross-attention: text queries attend to vision keys/values
    let scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();
    let cross_attn = b.add_attention(
        q,
        vision_k,
        vision_v,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );

    // Output projection + residual
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let attn_out = b.add_linear(cross_attn, out_w, None, &shape);
    let out = b.add_binary_add(text_input, attn_out, &shape);

    b.build(out).expect("valid cross-attention kernel")
}

/// Bindings for cross-modal attention.
fn cross_attention_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let vision_k = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);
    let vision_v = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                 // text_hidden
        TensorParamBinding::ConstantScalar(1e-5),     // norm_eps
        TensorParamBinding::ConstantTensor(norm_w),   // norm_weight
        TensorParamBinding::ConstantTensor(q_w),      // q_weight
        TensorParamBinding::ConstantTensor(vision_k), // vision_k
        TensorParamBinding::ConstantTensor(vision_v), // vision_v
        TensorParamBinding::ConstantTensor(out_w),    // out_weight
    ]
}

/// IBP bounds through cross-modal attention (text Q, vision K/V).
///
/// RMSNorm -> cross-attention (text attending to vision) -> residual.
#[test]
fn test_cross_attention_rmsnorm_ibp() {
    let def = build_cross_attention_kernel();
    let bindings = cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-modal attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "cross-attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attention RMSNorm IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through cross-modal attention.
///
/// Tests CROWN linearization through RMSNorm + softmax (cross-attention).
/// Uses IbpValidated mode.
#[test]
fn test_cross_attention_rmsnorm_crown() {
    let def = build_cross_attention_kernel();
    let bindings = cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attention RMSNorm: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 28. Granite Decoder Block with Self-Attention + Cross-Attention + SwiGLU
// ===========================================================================

/// Build a full Granite decoder block with self-attention + cross-attention + SwiGLU.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- text hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Architecture: RMSNorm -> self-attention -> residual -> RMSNorm ->
/// cross-attention (vision K/V constant) -> residual -> RMSNorm -> SwiGLU -> residual.
fn build_granite_decoder_self_cross_swiglu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_decoder_self_cross_swiglu");

    let input = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    // --- Self-attention ---
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    let sq_w = b.add_input("self_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sk_w = b.add_input("self_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sv_w = b.add_input("self_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let sout_w = b.add_input("self_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let sq = b.add_linear(normed1, sq_w, None, &shape);
    let sk = b.add_linear(normed1, sk_w, None, &shape);
    let sv = b.add_linear(normed1, sv_w, None, &shape);

    let self_attn = b.add_attention(sq, sk, sv, AttentionMask::Causal, Some(scale), &shape);
    let self_attn_out = b.add_linear(self_attn, sout_w, None, &shape);
    let res1 = b.add_binary_add(input, self_attn_out, &shape);

    // --- Cross-attention ---
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    let cq_w = b.add_input("cross_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let cq = b.add_linear(normed2, cq_w, None, &shape);

    let vision_k = b.add_input("vision_k", &[NUM_PATCHES, HIDDEN_DIM]);
    let vision_v = b.add_input("vision_v", &[NUM_PATCHES, HIDDEN_DIM]);

    let cross_attn = b.add_attention(
        cq,
        vision_k,
        vision_v,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );

    let cout_w = b.add_input("cross_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let cross_attn_out = b.add_linear(cross_attn, cout_w, None, &shape);
    let res2 = b.add_binary_add(res1, cross_attn_out, &shape);

    // --- SwiGLU FFN ---
    let norm3_eps = b.add_input("norm3_eps", &[1]);
    let norm3_w = b.add_input("norm3_weight", &[HIDDEN_DIM]);
    let normed3 = b.add_rms_norm(res2, norm3_eps, 1, norm3_w, &shape);

    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed3, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed3, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    let out = b.add_binary_add(res2, ffn_out, &shape);

    b.build(out)
        .expect("valid Granite decoder self+cross+SwiGLU kernel")
}

/// Bindings for Granite decoder block with self + cross attention + SwiGLU.
fn granite_decoder_self_cross_swiglu_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let vision_k = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);
    let vision_v = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // text_hidden
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()),  // self_q_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()),  // self_k_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()),  // self_v_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()),  // self_out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm2_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()),  // cross_q_weight
        TensorParamBinding::ConstantTensor(vision_k),       // vision_k
        TensorParamBinding::ConstantTensor(vision_v),       // vision_v
        TensorParamBinding::ConstantTensor(qkv_w),          // cross_out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm3_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm3_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
    ]
}

/// IBP bounds through Granite decoder with self + cross attention + SwiGLU.
///
/// Full decoder block: RMSNorm -> self-attn -> residual -> RMSNorm ->
/// cross-attn (vision K/V) -> residual -> RMSNorm -> SwiGLU -> residual.
#[test]
fn test_granite_decoder_self_cross_swiglu_ibp() {
    let def = build_granite_decoder_self_cross_swiglu_kernel();
    let bindings = granite_decoder_self_cross_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Granite decoder self+cross+SwiGLU");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "decoder self+cross+SwiGLU output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite decoder self+cross+SwiGLU IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through Granite decoder with self + cross attention + SwiGLU.
///
/// Tests CROWN through the deepest single-block architecture: 3x RMSNorm,
/// self-attention (causal), cross-attention (standard), SwiGLU (sigmoid +
/// McCormick), 3x residual. Uses IbpValidated mode.
#[test]
fn test_granite_decoder_self_cross_swiglu_crown() {
    let def = build_granite_decoder_self_cross_swiglu_kernel();
    let bindings = granite_decoder_self_cross_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite decoder self+cross+SwiGLU: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 29. Deep Decoder Stack: 4-Layer with Self + Cross Attention
// ===========================================================================

/// Build a 4-layer Granite decoder stack with self + cross attention at each layer.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- text hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Each layer: RMSNorm -> self-attn -> res -> RMSNorm -> cross-attn -> res ->
/// RMSNorm -> SwiGLU -> res. 4 layers deep.
fn build_granite_4layer_decoder_stack_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("granite_4layer_decoder_stack");

    let input = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer in 0..4 {
        let p = format!("l{layer}");

        // Self-attention
        let n1_eps = b.add_input(&format!("{p}_n1_eps"), &[1]);
        let n1_w = b.add_input(&format!("{p}_n1_weight"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, n1_eps, 1, n1_w, &shape);

        let sq_w = b.add_input(&format!("{p}_sq_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let sk_w = b.add_input(&format!("{p}_sk_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let sv_w = b.add_input(&format!("{p}_sv_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let so_w = b.add_input(&format!("{p}_so_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let sq = b.add_linear(normed1, sq_w, None, &shape);
        let sk = b.add_linear(normed1, sk_w, None, &shape);
        let sv = b.add_linear(normed1, sv_w, None, &shape);

        let sa = b.add_attention(sq, sk, sv, AttentionMask::Causal, Some(scale), &shape);
        let sa_out = b.add_linear(sa, so_w, None, &shape);
        let res1 = b.add_binary_add(current, sa_out, &shape);

        // Cross-attention
        let n2_eps = b.add_input(&format!("{p}_n2_eps"), &[1]);
        let n2_w = b.add_input(&format!("{p}_n2_weight"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

        let cq_w = b.add_input(&format!("{p}_cq_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let cq = b.add_linear(normed2, cq_w, None, &shape);

        let vk = b.add_input(&format!("{p}_vk"), &[NUM_PATCHES, HIDDEN_DIM]);
        let vv = b.add_input(&format!("{p}_vv"), &[NUM_PATCHES, HIDDEN_DIM]);

        let ca = b.add_attention(cq, vk, vv, AttentionMask::Standard, Some(scale), &shape);
        let co_w = b.add_input(&format!("{p}_co_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let ca_out = b.add_linear(ca, co_w, None, &shape);
        let res2 = b.add_binary_add(res1, ca_out, &shape);

        // SwiGLU FFN
        let n3_eps = b.add_input(&format!("{p}_n3_eps"), &[1]);
        let n3_w = b.add_input(&format!("{p}_n3_weight"), &[HIDDEN_DIM]);
        let normed3 = b.add_rms_norm(res2, n3_eps, 1, n3_w, &shape);

        let gw = b.add_input(&format!("{p}_gate_w"), &[FFN_DIM, HIDDEN_DIM]);
        let uw = b.add_input(&format!("{p}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
        let dw = b.add_input(&format!("{p}_down_w"), &[HIDDEN_DIM, FFN_DIM]);

        let gate = b.add_linear(normed3, gw, None, &ffn_shape);
        let gate_sig = b.add_sigmoid(gate, &ffn_shape);
        let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
        let up = b.add_linear(normed3, uw, None, &ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
        let ffn_out = b.add_linear(hidden, dw, None, &shape);

        current = b.add_binary_add(res2, ffn_out, &shape);
    }

    b.build(current)
        .expect("valid Granite 4-layer decoder stack kernel")
}

/// Bindings for 4-layer Granite decoder stack.
fn granite_4layer_decoder_stack_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let vision_k = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);
    let vision_v = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // text_hidden
    for _layer in 0..4 {
        // Self-attention
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        // Cross-attention
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(vision_k.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(vision_v.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        // SwiGLU FFN
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone()));
    }
    bindings
}

/// IBP bounds through 4-layer Granite decoder stack.
///
/// 4 layers x (self-attn + cross-attn + SwiGLU) = 12 attention/FFN blocks.
/// Tests whether IBP bounds remain finite through deep decoder stacks.
#[test]
fn test_granite_4layer_decoder_stack_ibp() {
    let def = build_granite_4layer_decoder_stack_kernel();
    let bindings = granite_4layer_decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-layer Granite decoder stack");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "4-layer decoder stack output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite 4-layer decoder stack IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through 4-layer Granite decoder stack.
///
/// Tests CROWN linearization depth through 4 layers of self-attention +
/// cross-attention + SwiGLU. Uses IbpValidated mode.
#[test]
fn test_granite_4layer_decoder_stack_crown() {
    let def = build_granite_4layer_decoder_stack_kernel();
    let bindings = granite_4layer_decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Granite 4-layer decoder stack: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 30. Patch Embedding at 64x64 Resolution
// ===========================================================================

/// Build patch embedding kernel at 64x64 resolution (16 patches).
///
/// Input: `[IN_CHANNELS, IMG_SIZE_64, IMG_SIZE_64]` (Variable).
/// Output: `[NUM_PATCHES_64, HIDDEN_DIM]`.
fn build_patch_embedding_64x64_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("patch_embedding_64x64");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE_64, IMG_SIZE_64]);
    let conv_w = b.add_input(
        "conv_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let conv_b = b.add_input("conv_bias", &[HIDDEN_DIM]);

    let conv_out_shape = [HIDDEN_DIM, GRID_SIZE_64, GRID_SIZE_64];
    let conv = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &conv_out_shape,
    );

    let flat_shape = [HIDDEN_DIM, NUM_PATCHES_64];
    let flat = b.add_reshape(conv, &flat_shape);
    let out = b.add_transpose(flat, &[1, 0], &[NUM_PATCHES_64, HIDDEN_DIM]);

    b.build(out).expect("valid patch embedding 64x64 kernel")
}

/// Bindings for 64x64 patch embedding.
fn patch_embedding_64x64_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,             // image
        TensorParamBinding::ConstantTensor(w),    // conv_weight
        TensorParamBinding::ConstantTensor(bias), // conv_bias
    ]
}

/// IBP bounds through patch embedding at 64x64 resolution (16 patches).
///
/// Tests that bound propagation scales correctly to 4x the patch count
/// vs the default 32x32 resolution.
#[test]
fn test_patch_embedding_64x64_ibp() {
    let def = build_patch_embedding_64x64_kernel();
    let bindings = patch_embedding_64x64_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE_64, IMG_SIZE_64]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 64x64 patch embedding");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES_64, HIDDEN_DIM],
        "64x64 patch embedding output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch embedding 64x64 IBP ({NUM_PATCHES_64} patches): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 31. Patch Embedding at 48x48 with Patch Size 8
// ===========================================================================

/// Build patch embedding kernel at 48x48, patch_size=8 (36 patches).
///
/// Input: `[IN_CHANNELS, IMG_SIZE_48, IMG_SIZE_48]` (Variable).
/// Output: `[NUM_PATCHES_48_P8, HIDDEN_DIM]`.
///
/// Tests bound propagation with a different patch size and higher patch count.
fn build_patch_embedding_48x48_p8_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("patch_embedding_48x48_p8");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE_48, IMG_SIZE_48]);
    let conv_w = b.add_input(
        "conv_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE_8, PATCH_SIZE_8],
    );
    let conv_b = b.add_input("conv_bias", &[HIDDEN_DIM]);

    let conv_out_shape = [HIDDEN_DIM, GRID_SIZE_48_P8, GRID_SIZE_48_P8];
    let conv = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE_8,
        PATCH_SIZE_8,
        0,
        0,
        &conv_out_shape,
    );

    let flat_shape = [HIDDEN_DIM, NUM_PATCHES_48_P8];
    let flat = b.add_reshape(conv, &flat_shape);
    let out = b.add_transpose(flat, &[1, 0], &[NUM_PATCHES_48_P8, HIDDEN_DIM]);

    b.build(out).expect("valid patch embedding 48x48 p8 kernel")
}

/// Bindings for 48x48 patch_size=8 patch embedding.
fn patch_embedding_48x48_p8_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE_8, PATCH_SIZE_8]),
        WEIGHT_MAG,
    );
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,             // image
        TensorParamBinding::ConstantTensor(w),    // conv_weight
        TensorParamBinding::ConstantTensor(bias), // conv_bias
    ]
}

/// IBP bounds through patch embedding at 48x48, patch_size=8 (36 patches).
///
/// Tests bound propagation with higher spatial resolution (6x6 grid, 36 patches).
#[test]
fn test_patch_embedding_48x48_p8_ibp() {
    let def = build_patch_embedding_48x48_p8_kernel();
    let bindings = patch_embedding_48x48_p8_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE_48, IMG_SIZE_48]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 48x48 p8 patch embedding");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES_48_P8, HIDDEN_DIM],
        "48x48 p8 patch embedding output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Patch embedding 48x48 p8 IBP ({NUM_PATCHES_48_P8} patches): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 32. End-to-End VLM Pipeline: 2 Encoder + 2 Decoder Layers
// ===========================================================================

/// Build end-to-end VLM pipeline: patch embed -> 2 ViT encoder -> projection
/// -> 2 Granite decoder (self+cross+SwiGLU).
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable -- image pixels).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// This is the most comprehensive pipeline test, covering vision encoding,
/// projection, and multimodal decoding.
fn build_vlm_2enc_2dec_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("vlm_2enc_2dec");

    // --- Patch embedding ---
    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let pe_conv_w = b.add_input(
        "pe_conv_w",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let pe_conv_b = b.add_input("pe_conv_b", &[HIDDEN_DIM]);

    let conv_out_shape = [HIDDEN_DIM, GRID_SIZE, GRID_SIZE];
    let conv = b.add_conv2d(
        image,
        pe_conv_w,
        Some(pe_conv_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &conv_out_shape,
    );
    let flat_shape = [HIDDEN_DIM, NUM_PATCHES];
    let flat = b.add_reshape(conv, &flat_shape);
    let patches = b.add_transpose(flat, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);

    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];
    let patch_ffn_shape = [NUM_PATCHES, FFN_DIM];
    let vis_scale = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();

    // --- 2 SigLIP2 encoder layers ---
    let mut vis_current = patches;
    for layer in 0..2 {
        let p = format!("ve{layer}");

        let ln1_eps = b.add_input(&format!("{p}_ln1_eps"), &[1]);
        let ln1_w = b.add_input(&format!("{p}_ln1_w"), &[HIDDEN_DIM]);
        let ln1_b = b.add_input(&format!("{p}_ln1_b"), &[HIDDEN_DIM]);
        let normed = b.add_layer_norm(vis_current, ln1_eps, 1, ln1_w, ln1_b, &patch_shape);

        let qw = b.add_input(&format!("{p}_qw"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let kw = b.add_input(&format!("{p}_kw"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let vw = b.add_input(&format!("{p}_vw"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let ow = b.add_input(&format!("{p}_ow"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed, qw, None, &patch_shape);
        let k = b.add_linear(normed, kw, None, &patch_shape);
        let v = b.add_linear(normed, vw, None, &patch_shape);

        let attn = b.add_attention(
            q,
            k,
            v,
            AttentionMask::Standard,
            Some(vis_scale),
            &patch_shape,
        );
        let attn_out = b.add_linear(attn, ow, None, &patch_shape);
        let res1 = b.add_binary_add(vis_current, attn_out, &patch_shape);

        let ln2_eps = b.add_input(&format!("{p}_ln2_eps"), &[1]);
        let ln2_w = b.add_input(&format!("{p}_ln2_w"), &[HIDDEN_DIM]);
        let ln2_b = b.add_input(&format!("{p}_ln2_b"), &[HIDDEN_DIM]);
        let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &patch_shape);

        let fc1w = b.add_input(&format!("{p}_fc1w"), &[FFN_DIM, HIDDEN_DIM]);
        let fc2w = b.add_input(&format!("{p}_fc2w"), &[HIDDEN_DIM, FFN_DIM]);

        let h = b.add_linear(normed2, fc1w, None, &patch_ffn_shape);
        let h = b.add_gelu(h, &patch_ffn_shape);
        let ffn_out = b.add_linear(h, fc2w, None, &patch_shape);
        vis_current = b.add_binary_add(res1, ffn_out, &patch_shape);
    }

    // --- Vision projection ---
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vision_proj = b.add_linear(vis_current, proj_w, None, &patch_shape);

    // --- 2 Granite decoder layers (text input is constant for this test) ---
    let text_shape = [SEQ_LEN, HIDDEN_DIM];
    let text_ffn_shape = [SEQ_LEN, FFN_DIM];
    let dec_scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    // Start with a constant text embedding (we're verifying vision -> output path)
    // Use a linear from vision_proj to get text-shaped input for the decoder
    let text_proj_w = b.add_input("text_proj_w", &[SEQ_LEN, NUM_PATCHES]);
    let vision_transposed = b.add_transpose(vision_proj, &[1, 0], &[HIDDEN_DIM, NUM_PATCHES]);
    let text_input = b.add_linear(vision_transposed, text_proj_w, None, &[HIDDEN_DIM, SEQ_LEN]);
    let mut dec_current = b.add_transpose(text_input, &[1, 0], &text_shape);

    for layer in 0..2 {
        let p = format!("d{layer}");

        // Self-attention
        let n1_eps = b.add_input(&format!("{p}_n1_eps"), &[1]);
        let n1_w = b.add_input(&format!("{p}_n1_w"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(dec_current, n1_eps, 1, n1_w, &text_shape);

        let sq = b.add_input(&format!("{p}_sq"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let sk = b.add_input(&format!("{p}_sk"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let sv = b.add_input(&format!("{p}_sv"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let so = b.add_input(&format!("{p}_so"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, sq, None, &text_shape);
        let k = b.add_linear(normed1, sk, None, &text_shape);
        let v = b.add_linear(normed1, sv, None, &text_shape);

        let sa = b.add_attention(q, k, v, AttentionMask::Causal, Some(dec_scale), &text_shape);
        let sa_out = b.add_linear(sa, so, None, &text_shape);
        let res1 = b.add_binary_add(dec_current, sa_out, &text_shape);

        // SwiGLU FFN (skip cross-attention for graph simplicity)
        let n3_eps = b.add_input(&format!("{p}_n3_eps"), &[1]);
        let n3_w = b.add_input(&format!("{p}_n3_w"), &[HIDDEN_DIM]);
        let normed3 = b.add_rms_norm(res1, n3_eps, 1, n3_w, &text_shape);

        let gw = b.add_input(&format!("{p}_gw"), &[FFN_DIM, HIDDEN_DIM]);
        let uw = b.add_input(&format!("{p}_uw"), &[FFN_DIM, HIDDEN_DIM]);
        let dw = b.add_input(&format!("{p}_dw"), &[HIDDEN_DIM, FFN_DIM]);

        let gate = b.add_linear(normed3, gw, None, &text_ffn_shape);
        let gate_sig = b.add_sigmoid(gate, &text_ffn_shape);
        let gate_act = b.add_binary_mul(gate, gate_sig, &text_ffn_shape);
        let up = b.add_linear(normed3, uw, None, &text_ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &text_ffn_shape);
        let ffn_out = b.add_linear(hidden, dw, None, &text_shape);

        dec_current = b.add_binary_add(res1, ffn_out, &text_shape);
    }

    b.build(dec_current).expect("valid VLM 2-enc 2-dec kernel")
}

/// Bindings for end-to-end VLM pipeline (2 encoder + 2 decoder).
fn vlm_2enc_2dec_bindings() -> Vec<TensorParamBinding> {
    let pe_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let pe_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let text_proj_w = ArrayD::from_elem(IxDyn(&[SEQ_LEN, NUM_PATCHES]), WEIGHT_MAG);

    let mut bindings = vec![
        TensorParamBinding::Variable,             // image
        TensorParamBinding::ConstantTensor(pe_w), // pe_conv_w
        TensorParamBinding::ConstantTensor(pe_b), // pe_conv_b
    ];

    // 2 encoder layers
    for _layer in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln1_eps
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // ln1_w
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // ln1_b
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // qw
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // kw
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // vw
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // ow
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // ln2_eps
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // ln2_w
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // ln2_b
        bindings.push(TensorParamBinding::ConstantTensor(fc1_w.clone())); // fc1w
        bindings.push(TensorParamBinding::ConstantTensor(fc2_w.clone())); // fc2w
    }

    // Vision projection
    bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone())); // proj_w
                                                                      // Text projection
    bindings.push(TensorParamBinding::ConstantTensor(text_proj_w)); // text_proj_w

    // 2 decoder layers
    for _layer in 0..2 {
        // Self-attention
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkv_w.clone()));
        // SwiGLU FFN
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone()));
    }
    bindings
}

/// IBP bounds through end-to-end VLM: patch embed -> 2 ViT encoder ->
/// projection -> 2 Granite decoder with SwiGLU.
///
/// The deepest end-to-end pipeline test, covering the full vision-language
/// model from image pixels to decoded text hidden states.
#[test]
fn test_vlm_2enc_2dec_ibp() {
    let def = build_vlm_2enc_2dec_kernel();
    let bindings = vlm_2enc_2dec_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through VLM 2-enc 2-dec pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "VLM 2-enc 2-dec output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("VLM 2-enc 2-dec IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 33. RMSNorm 6x Chain (Accumulation Test)
// ===========================================================================

/// Build a chain of 6 sequential RMSNorm layers.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Tests whether RMSNorm accumulates or dampens bounds over many
/// sequential applications. Each RMSNorm divides by sqrt(mean(x^2) + eps),
/// which should keep bounds stable.
fn build_rmsnorm_6x_chain_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("rmsnorm_6x_chain");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut current = input;
    for i in 0..6 {
        let eps = b.add_input(&format!("eps_{i}"), &[1]);
        let w = b.add_input(&format!("weight_{i}"), &[HIDDEN_DIM]);
        current = b.add_rms_norm(current, eps, 1, w, &shape);
    }

    b.build(current).expect("valid RMSNorm 6x chain kernel")
}

/// Bindings for RMSNorm 6x chain.
fn rmsnorm_6x_chain_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
    for _i in 0..6 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
        bindings.push(TensorParamBinding::ConstantTensor(w.clone())); // weight
    }
    bindings
}

/// IBP bounds through 6 sequential RMSNorm layers.
///
/// Tests bound accumulation through deep normalization chains.
/// RMSNorm should not amplify bounds -- each application normalizes.
#[test]
fn test_rmsnorm_6x_chain_ibp() {
    let def = build_rmsnorm_6x_chain_kernel();
    let bindings = rmsnorm_6x_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through RMSNorm 6x chain");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "RMSNorm 6x chain output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RMSNorm 6x chain IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through 6 sequential RMSNorm layers.
///
/// Tests CROWN linearization stability through deep normalization chains.
/// Uses IbpValidated mode.
#[test]
fn test_rmsnorm_6x_chain_crown() {
    let def = build_rmsnorm_6x_chain_kernel();
    let bindings = rmsnorm_6x_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RMSNorm 6x chain: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 34. LayerNorm 6x Chain (Accumulation Test)
// ===========================================================================

/// Build a chain of 6 sequential LayerNorm layers.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Parallel test to RMSNorm 6x chain for LayerNorm. Tests whether
/// LayerNorm accumulates bounds differently than RMSNorm over deep chains.
fn build_layernorm_6x_chain_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("layernorm_6x_chain");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut current = input;
    for i in 0..6 {
        let eps = b.add_input(&format!("eps_{i}"), &[1]);
        let w = b.add_input(&format!("weight_{i}"), &[HIDDEN_DIM]);
        let bias = b.add_input(&format!("bias_{i}"), &[HIDDEN_DIM]);
        current = b.add_layer_norm(current, eps, 1, w, bias, &shape);
    }

    b.build(current).expect("valid LayerNorm 6x chain kernel")
}

/// Bindings for LayerNorm 6x chain.
fn layernorm_6x_chain_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
    for _i in 0..6 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // eps
        bindings.push(TensorParamBinding::ConstantTensor(w.clone())); // weight
        bindings.push(TensorParamBinding::ConstantTensor(bias.clone())); // bias
    }
    bindings
}

/// IBP bounds through 6 sequential LayerNorm layers.
///
/// Tests bound accumulation through deep LayerNorm chains.
/// Parallel to RMSNorm 6x chain for comparison.
#[test]
fn test_layernorm_6x_chain_ibp() {
    let def = build_layernorm_6x_chain_kernel();
    let bindings = layernorm_6x_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through LayerNorm 6x chain");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "LayerNorm 6x chain output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("LayerNorm 6x chain IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds through 6 sequential LayerNorm layers.
///
/// Tests CROWN linearization stability through deep LayerNorm chains.
/// Uses IbpValidated mode.
#[test]
fn test_layernorm_6x_chain_crown() {
    let def = build_layernorm_6x_chain_kernel();
    let bindings = layernorm_6x_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("LayerNorm 6x chain: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 35. Full cross-attention with RMSNorm pre/post
// ===========================================================================

/// Build cross-attention with RMSNorm both before and after the attention block.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- text hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Architecture: RMSNorm(pre) -> Q projection -> cross-attn(vision K/V) ->
/// output projection -> residual -> RMSNorm(post).
fn build_cross_attn_rmsnorm_pre_post_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("cross_attn_rmsnorm_pre_post");

    let input = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let pre_eps = b.add_input("pre_norm_eps", &[1]);
    let pre_w = b.add_input("pre_norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, pre_eps, 1, pre_w, &shape);

    // Q from text, K/V from vision (constant)
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let q = b.add_linear(normed, q_w, None, &shape);

    let vision_k = b.add_input("vision_k", &[NUM_PATCHES, HIDDEN_DIM]);
    let vision_v = b.add_input("vision_v", &[NUM_PATCHES, HIDDEN_DIM]);

    let cross_attn = b.add_attention(
        q,
        vision_k,
        vision_v,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );

    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let attn_out = b.add_linear(cross_attn, out_w, None, &shape);
    let res = b.add_binary_add(input, attn_out, &shape);

    // Post-attention RMSNorm
    let post_eps = b.add_input("post_norm_eps", &[1]);
    let post_w = b.add_input("post_norm_weight", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(res, post_eps, 1, post_w, &shape);

    b.build(out)
        .expect("valid cross-attn with RMSNorm pre/post kernel")
}

/// Bindings for cross-attention with RMSNorm pre/post.
fn cross_attn_rmsnorm_pre_post_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let vision_k = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);
    let vision_v = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // text_hidden
        TensorParamBinding::ConstantScalar(1e-5),           // pre_norm_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // pre_norm_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(vision_k),       // vision_k
        TensorParamBinding::ConstantTensor(vision_v),       // vision_v
        TensorParamBinding::ConstantTensor(attn_w),         // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // post_norm_eps
        TensorParamBinding::ConstantTensor(norm_w),         // post_norm_weight
    ]
}

/// IBP through cross-attention with RMSNorm pre and post.
///
/// Tests bounds through full pre-norm -> cross-attn -> residual -> post-norm
/// pattern used in Granite-Docling VLM decoder layers.
#[test]
fn test_cross_attn_rmsnorm_pre_post_ibp() {
    let def = build_cross_attn_rmsnorm_pre_post_kernel();
    let bindings = cross_attn_rmsnorm_pre_post_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attn RMSNorm pre/post");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "cross-attn RMSNorm pre/post output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attn RMSNorm pre/post IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through cross-attention with RMSNorm pre and post.
///
/// Tests CROWN linearization through both RMSNorm layers + softmax.
#[test]
fn test_cross_attn_rmsnorm_pre_post_crown() {
    let def = build_cross_attn_rmsnorm_pre_post_kernel();
    let bindings = cross_attn_rmsnorm_pre_post_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attn RMSNorm pre/post: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 36. Multi-head cross-attention: 8-head variant
// ===========================================================================

/// Build 8-head cross-attention with per-head Q/K/V projections.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- text hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Uses 8 attention heads (HEAD_DIM=8 per head for HIDDEN_DIM=64).
/// Pre-norm RMSNorm -> Q/K/V project -> attention -> output project -> residual.
fn build_multihead_cross_attn_8h_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("multihead_cross_attn_8h");

    let input = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let num_heads_8: usize = 8;
    let head_dim_8 = HIDDEN_DIM / num_heads_8; // 8
    let scale = 1.0 / (head_dim_8 as f32).sqrt();

    // RMSNorm
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, norm_eps, 1, norm_w, &shape);

    // Full Q/K/V projections (multi-head packed into HIDDEN_DIM)
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);

    // K/V from frozen vision features
    let vision_k = b.add_input("vision_k", &[NUM_PATCHES, HIDDEN_DIM]);
    let vision_v = b.add_input("vision_v", &[NUM_PATCHES, HIDDEN_DIM]);

    let k = b.add_linear(vision_k, k_w, None, &[NUM_PATCHES, HIDDEN_DIM]);
    let v = b.add_linear(vision_v, v_w, None, &[NUM_PATCHES, HIDDEN_DIM]);

    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(input, attn_out, &shape);

    b.build(out).expect("valid 8-head cross-attention kernel")
}

/// Bindings for 8-head cross-attention.
fn multihead_cross_attn_8h_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let vision_feat = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                            // text_hidden
        TensorParamBinding::ConstantScalar(1e-5),                // norm_eps
        TensorParamBinding::ConstantTensor(norm_w),              // norm_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()),      // q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()),      // k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()),      // v_weight
        TensorParamBinding::ConstantTensor(attn_w),              // out_weight
        TensorParamBinding::ConstantTensor(vision_feat.clone()), // vision_k
        TensorParamBinding::ConstantTensor(vision_feat),         // vision_v
    ]
}

/// IBP through 8-head cross-attention.
///
/// Tests bounds propagation with a wider attention head configuration (8 heads).
#[test]
fn test_multihead_cross_attn_8h_ibp() {
    let def = build_multihead_cross_attn_8h_kernel();
    let bindings = multihead_cross_attn_8h_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 8-head cross-attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "8-head cross-attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("8-head cross-attention IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 37. Multi-head cross-attention: 16-head variant
// ===========================================================================

/// Build 16-head cross-attention with many small heads.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- text hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Uses 16 attention heads (HEAD_DIM=4 per head for HIDDEN_DIM=64).
/// Stress-tests bounds propagation with many small heads.
fn build_multihead_cross_attn_16h_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("multihead_cross_attn_16h");

    let input = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let num_heads_16: usize = 16;
    let head_dim_16 = HIDDEN_DIM / num_heads_16; // 4
    let scale = 1.0 / (head_dim_16 as f32).sqrt();

    // RMSNorm
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, norm_eps, 1, norm_w, &shape);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);

    // Frozen vision K/V
    let vision_k = b.add_input("vision_k", &[NUM_PATCHES, HIDDEN_DIM]);
    let vision_v = b.add_input("vision_v", &[NUM_PATCHES, HIDDEN_DIM]);

    let attn = b.add_attention(
        q,
        vision_k,
        vision_v,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(input, attn_out, &shape);

    b.build(out).expect("valid 16-head cross-attention kernel")
}

/// Bindings for 16-head cross-attention.
fn multihead_cross_attn_16h_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let vision_feat = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                            // text_hidden
        TensorParamBinding::ConstantScalar(1e-5),                // norm_eps
        TensorParamBinding::ConstantTensor(norm_w),              // norm_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()),      // q_weight
        TensorParamBinding::ConstantTensor(attn_w),              // out_weight
        TensorParamBinding::ConstantTensor(vision_feat.clone()), // vision_k
        TensorParamBinding::ConstantTensor(vision_feat),         // vision_v
    ]
}

/// IBP through 16-head cross-attention.
///
/// Many small attention heads (4 dimensions each) test whether bounds
/// propagation scales with head count.
#[test]
fn test_multihead_cross_attn_16h_ibp() {
    let def = build_multihead_cross_attn_16h_kernel();
    let bindings = multihead_cross_attn_16h_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 16-head cross-attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "16-head cross-attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("16-head cross-attention IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through 16-head cross-attention.
///
/// Tests CROWN linearization with many small heads.
#[test]
fn test_multihead_cross_attn_16h_crown() {
    let def = build_multihead_cross_attn_16h_kernel();
    let bindings = multihead_cross_attn_16h_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("16-head cross-attn: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 38. Cross-attention with KV-cache: cached vision features maintain bounds
// ===========================================================================

/// Build cross-attention simulating KV-cache with extended vision sequence.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- current text token hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Models KV-cache inference: vision K/V have been pre-computed and cached
/// over a longer sequence (2x NUM_PATCHES). Text queries attend to the full
/// cached vision features. Verifies bounds are maintained when KV sequence
/// is longer than Q sequence.
fn build_cross_attn_kv_cache_kernel() -> TensorKernelDef {
    let kv_len = NUM_PATCHES * 2; // cached vision features (extended)
    let mut b = TensorBlockBuilder::new("cross_attn_kv_cache");

    let input = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    // RMSNorm
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, norm_eps, 1, norm_w, &shape);

    // Q from text
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let q = b.add_linear(normed, q_w, None, &shape);

    // Cached vision K/V (longer sequence, pre-computed)
    let vision_k = b.add_input("cached_vision_k", &[kv_len, HIDDEN_DIM]);
    let vision_v = b.add_input("cached_vision_v", &[kv_len, HIDDEN_DIM]);

    // Cross-attention: Q=[SEQ_LEN, D], K/V=[kv_len, D]
    let attn = b.add_attention(
        q,
        vision_k,
        vision_v,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );

    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(input, attn_out, &shape);

    b.build(out).expect("valid cross-attn KV-cache kernel")
}

/// Bindings for cross-attention with KV-cache.
fn cross_attn_kv_cache_bindings() -> Vec<TensorParamBinding> {
    let kv_len = NUM_PATCHES * 2;
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let vision_k = ArrayD::from_elem(IxDyn(&[kv_len, HIDDEN_DIM]), WEIGHT_MAG);
    let vision_v = ArrayD::from_elem(IxDyn(&[kv_len, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // text_hidden
        TensorParamBinding::ConstantScalar(1e-5),           // norm_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(vision_k),       // cached_vision_k
        TensorParamBinding::ConstantTensor(vision_v),       // cached_vision_v
        TensorParamBinding::ConstantTensor(attn_w),         // out_weight
    ]
}

/// IBP through cross-attention with extended KV-cache.
///
/// Verifies that bounds remain valid when KV sequence (2x NUM_PATCHES)
/// is longer than Q sequence (SEQ_LEN). Models cached vision features
/// in autoregressive decoding.
#[test]
fn test_cross_attn_kv_cache_ibp() {
    let def = build_cross_attn_kv_cache_kernel();
    let bindings = cross_attn_kv_cache_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attn KV-cache");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "KV-cache cross-attn output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Cross-attn KV-cache IBP (kv_len={}): bounds=[{lo_min}, {hi_max}]",
        NUM_PATCHES * 2
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through cross-attention with extended KV-cache.
#[test]
fn test_cross_attn_kv_cache_crown() {
    let def = build_cross_attn_kv_cache_kernel();
    let bindings = cross_attn_kv_cache_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attn KV-cache CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 39. 4-layer stacked cross-attention blocks
// ===========================================================================

/// Build 4-layer stacked cross-attention blocks.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- text hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Each layer: RMSNorm -> cross-attention(vision K/V) -> residual.
/// Tests CROWN depth through repeated cross-attention + normalization.
fn build_stacked_cross_attn_4layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("stacked_cross_attn_4layer");

    let mut hidden = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    for i in 0..4 {
        let name = format!("layer{i}");

        // RMSNorm
        let norm_eps = b.add_input(&format!("{name}_norm_eps"), &[1]);
        let norm_w = b.add_input(&format!("{name}_norm_weight"), &[HIDDEN_DIM]);
        let normed = b.add_rms_norm(hidden, norm_eps, 1, norm_w, &shape);

        // Q from text
        let q_w = b.add_input(&format!("{name}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let q = b.add_linear(normed, q_w, None, &shape);

        // Vision K/V (frozen)
        let vk = b.add_input(&format!("{name}_vision_k"), &[NUM_PATCHES, HIDDEN_DIM]);
        let vv = b.add_input(&format!("{name}_vision_v"), &[NUM_PATCHES, HIDDEN_DIM]);

        let attn = b.add_attention(q, vk, vv, AttentionMask::Standard, Some(scale), &shape);

        let out_w = b.add_input(&format!("{name}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        hidden = b.add_binary_add(hidden, attn_out, &shape);
    }

    b.build(hidden)
        .expect("valid 4-layer stacked cross-attention kernel")
}

/// Bindings for 4-layer stacked cross-attention.
fn stacked_cross_attn_4layer_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let vision_feat = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // text_hidden

    for _i in 0..4 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm_weight
        bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(vision_feat.clone())); // vision_k
        bindings.push(TensorParamBinding::ConstantTensor(vision_feat.clone())); // vision_v
        bindings.push(TensorParamBinding::ConstantTensor(attn_w.clone())); // out_weight
    }

    bindings
}

/// IBP through 4-layer stacked cross-attention.
///
/// Tests bound accumulation through deep repeated cross-attention.
/// Verifies bounds remain finite through 4 attention + normalization layers.
#[test]
fn test_stacked_cross_attn_4layer_ibp() {
    let def = build_stacked_cross_attn_4layer_kernel();
    let bindings = stacked_cross_attn_4layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-layer stacked cross-attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "4-layer stacked cross-attn output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("4-layer stacked cross-attn IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through 4-layer stacked cross-attention.
///
/// Deepest CROWN test for cross-attention patterns: 4 layers of
/// RMSNorm + attention + residual.
#[test]
fn test_stacked_cross_attn_4layer_crown() {
    let def = build_stacked_cross_attn_4layer_kernel();
    let bindings = stacked_cross_attn_4layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("4-layer stacked cross-attn CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 40. Document image encoder pipeline
// ===========================================================================

/// Build a document image encoder pipeline: Conv2d patch embedding ->
/// reshape -> transpose -> positional add -> 2 transformer encoder blocks.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image pixels [0, 1]).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
///
/// Models a ResNet/ViT-style backbone: patch embedding extracts features,
/// positional encoding adds learned position information, then transformer
/// encoder blocks refine the representation.
fn build_doc_image_encoder_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("doc_image_encoder_pipeline");

    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];
    let ffn_shape = [NUM_PATCHES, FFN_DIM];
    let scale = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();

    // --- Patch embedding (Conv2d -> reshape -> transpose) ---
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        image,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &patch_shape);

    // --- Positional encoding (learned, added to patch embeddings) ---
    let pos_embed = b.add_input("pos_embed", &[NUM_PATCHES, HIDDEN_DIM]);
    let positioned = b.add_binary_add(patches, pos_embed, &patch_shape);

    // --- Encoder block 1: LayerNorm -> attention -> residual -> LN -> GELU MLP -> residual ---
    let ln1a_eps = b.add_input("enc1_ln1_eps", &[1]);
    let ln1a_w = b.add_input("enc1_ln1_weight", &[HIDDEN_DIM]);
    let ln1a_b = b.add_input("enc1_ln1_bias", &[HIDDEN_DIM]);
    let normed1a = b.add_layer_norm(positioned, ln1a_eps, 1, ln1a_w, ln1a_b, &patch_shape);

    let q1_w = b.add_input("enc1_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k1_w = b.add_input("enc1_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v1_w = b.add_input("enc1_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out1_w = b.add_input("enc1_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q1 = b.add_linear(normed1a, q1_w, None, &patch_shape);
    let k1 = b.add_linear(normed1a, k1_w, None, &patch_shape);
    let v1 = b.add_linear(normed1a, v1_w, None, &patch_shape);
    let attn1 = b.add_attention(
        q1,
        k1,
        v1,
        AttentionMask::Standard,
        Some(scale),
        &patch_shape,
    );
    let attn1_out = b.add_linear(attn1, out1_w, None, &patch_shape);
    let res1a = b.add_binary_add(positioned, attn1_out, &patch_shape);

    let ln1b_eps = b.add_input("enc1_ln2_eps", &[1]);
    let ln1b_w = b.add_input("enc1_ln2_weight", &[HIDDEN_DIM]);
    let ln1b_b = b.add_input("enc1_ln2_bias", &[HIDDEN_DIM]);
    let normed1b = b.add_layer_norm(res1a, ln1b_eps, 1, ln1b_w, ln1b_b, &patch_shape);

    let fc1a_w = b.add_input("enc1_fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc1b_w = b.add_input("enc1_fc2_weight", &[HIDDEN_DIM, FFN_DIM]);
    let h1 = b.add_linear(normed1b, fc1a_w, None, &ffn_shape);
    let h1 = b.add_gelu(h1, &ffn_shape);
    let ffn1_out = b.add_linear(h1, fc1b_w, None, &patch_shape);
    let enc1_out = b.add_binary_add(res1a, ffn1_out, &patch_shape);

    // --- Encoder block 2 ---
    let ln2a_eps = b.add_input("enc2_ln1_eps", &[1]);
    let ln2a_w = b.add_input("enc2_ln1_weight", &[HIDDEN_DIM]);
    let ln2a_b = b.add_input("enc2_ln1_bias", &[HIDDEN_DIM]);
    let normed2a = b.add_layer_norm(enc1_out, ln2a_eps, 1, ln2a_w, ln2a_b, &patch_shape);

    let q2_w = b.add_input("enc2_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k2_w = b.add_input("enc2_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v2_w = b.add_input("enc2_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out2_w = b.add_input("enc2_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q2 = b.add_linear(normed2a, q2_w, None, &patch_shape);
    let k2 = b.add_linear(normed2a, k2_w, None, &patch_shape);
    let v2 = b.add_linear(normed2a, v2_w, None, &patch_shape);
    let attn2 = b.add_attention(
        q2,
        k2,
        v2,
        AttentionMask::Standard,
        Some(scale),
        &patch_shape,
    );
    let attn2_out = b.add_linear(attn2, out2_w, None, &patch_shape);
    let res2a = b.add_binary_add(enc1_out, attn2_out, &patch_shape);

    let ln2b_eps = b.add_input("enc2_ln2_eps", &[1]);
    let ln2b_w = b.add_input("enc2_ln2_weight", &[HIDDEN_DIM]);
    let ln2b_b = b.add_input("enc2_ln2_bias", &[HIDDEN_DIM]);
    let normed2b = b.add_layer_norm(res2a, ln2b_eps, 1, ln2b_w, ln2b_b, &patch_shape);

    let fc2a_w = b.add_input("enc2_fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc2b_w = b.add_input("enc2_fc2_weight", &[HIDDEN_DIM, FFN_DIM]);
    let h2 = b.add_linear(normed2b, fc2a_w, None, &ffn_shape);
    let h2 = b.add_gelu(h2, &ffn_shape);
    let ffn2_out = b.add_linear(h2, fc2b_w, None, &patch_shape);
    let out = b.add_binary_add(res2a, ffn2_out, &patch_shape);

    b.build(out)
        .expect("valid doc image encoder pipeline kernel")
}

/// Bindings for document image encoder pipeline.
fn doc_image_encoder_pipeline_bindings() -> Vec<TensorParamBinding> {
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let pos_embed = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                  // image
        TensorParamBinding::ConstantTensor(patch_w),   // patch_weight
        TensorParamBinding::ConstantTensor(patch_b),   // patch_bias
        TensorParamBinding::ConstantTensor(pos_embed), // pos_embed
        // Encoder block 1
        TensorParamBinding::ConstantScalar(1e-5), // enc1_ln1_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // enc1_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // enc1_ln1_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc1_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc1_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc1_v_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc1_out_weight
        TensorParamBinding::ConstantScalar(1e-5), // enc1_ln2_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // enc1_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // enc1_ln2_bias
        TensorParamBinding::ConstantTensor(fc1_w.clone()), // enc1_fc1_weight
        TensorParamBinding::ConstantTensor(fc2_w.clone()), // enc1_fc2_weight
        // Encoder block 2
        TensorParamBinding::ConstantScalar(1e-5), // enc2_ln1_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // enc2_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // enc2_ln1_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc2_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc2_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc2_v_weight
        TensorParamBinding::ConstantTensor(attn_w), // enc2_out_weight
        TensorParamBinding::ConstantScalar(1e-5), // enc2_ln2_eps
        TensorParamBinding::ConstantTensor(ln_w), // enc2_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b), // enc2_ln2_bias
        TensorParamBinding::ConstantTensor(fc1_w), // enc2_fc1_weight
        TensorParamBinding::ConstantTensor(fc2_w), // enc2_fc2_weight
    ]
}

/// IBP through document image encoder pipeline.
///
/// Conv2d patch embedding -> positional encoding -> 2 transformer encoder blocks.
/// Verifies bounds propagation through the full image encoding pathway.
#[test]
fn test_compose_granite_docling_doc_image_encoder_ibp() {
    let def = build_doc_image_encoder_pipeline_kernel();
    let bindings = doc_image_encoder_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through doc image encoder pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "doc image encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Doc image encoder pipeline IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through document image encoder pipeline.
///
/// Tests CROWN linearization depth through Conv2d -> positional add ->
/// 2 full transformer encoder blocks with LayerNorm, attention, and GELU MLP.
#[test]
fn test_compose_granite_docling_doc_image_encoder_crown() {
    let def = build_doc_image_encoder_pipeline_kernel();
    let bindings = doc_image_encoder_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Doc image encoder pipeline: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 41. Text token encoder pipeline
// ===========================================================================

/// Build a text token encoder pipeline: embedding -> positional add ->
/// 2 transformer encoder blocks (LayerNorm -> attention -> GELU MLP).
///
/// Input: `[SEQ_LEN]` (Variable, integer token indices in [0, VOCAB_SIZE-1]).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Models the text side of a document understanding model: token embedding
/// lookup, learned positional encoding, then standard transformer encoder.
fn build_text_token_encoder_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("text_token_encoder_pipeline");

    let input = b.add_input("token_ids", &[SEQ_LEN]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();

    // --- Token embedding ---
    let embed_w = b.add_input("embed_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let embedded = b.add_embedding(input, embed_w, &shape);

    // --- Positional encoding (learned, added to embeddings) ---
    let pos_embed = b.add_input("pos_embed", &[SEQ_LEN, HIDDEN_DIM]);
    let positioned = b.add_binary_add(embedded, pos_embed, &shape);

    // --- Encoder block 1 ---
    let ln1a_eps = b.add_input("enc1_ln1_eps", &[1]);
    let ln1a_w = b.add_input("enc1_ln1_weight", &[HIDDEN_DIM]);
    let ln1a_b = b.add_input("enc1_ln1_bias", &[HIDDEN_DIM]);
    let normed1a = b.add_layer_norm(positioned, ln1a_eps, 1, ln1a_w, ln1a_b, &shape);

    let q1_w = b.add_input("enc1_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k1_w = b.add_input("enc1_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v1_w = b.add_input("enc1_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out1_w = b.add_input("enc1_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q1 = b.add_linear(normed1a, q1_w, None, &shape);
    let k1 = b.add_linear(normed1a, k1_w, None, &shape);
    let v1 = b.add_linear(normed1a, v1_w, None, &shape);
    let attn1 = b.add_attention(q1, k1, v1, AttentionMask::Standard, Some(scale), &shape);
    let attn1_out = b.add_linear(attn1, out1_w, None, &shape);
    let res1a = b.add_binary_add(positioned, attn1_out, &shape);

    let ln1b_eps = b.add_input("enc1_ln2_eps", &[1]);
    let ln1b_w = b.add_input("enc1_ln2_weight", &[HIDDEN_DIM]);
    let ln1b_b = b.add_input("enc1_ln2_bias", &[HIDDEN_DIM]);
    let normed1b = b.add_layer_norm(res1a, ln1b_eps, 1, ln1b_w, ln1b_b, &shape);

    let fc1a_w = b.add_input("enc1_fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc1b_w = b.add_input("enc1_fc2_weight", &[HIDDEN_DIM, FFN_DIM]);
    let h1 = b.add_linear(normed1b, fc1a_w, None, &ffn_shape);
    let h1 = b.add_gelu(h1, &ffn_shape);
    let ffn1_out = b.add_linear(h1, fc1b_w, None, &shape);
    let enc1_out = b.add_binary_add(res1a, ffn1_out, &shape);

    // --- Encoder block 2 ---
    let ln2a_eps = b.add_input("enc2_ln1_eps", &[1]);
    let ln2a_w = b.add_input("enc2_ln1_weight", &[HIDDEN_DIM]);
    let ln2a_b = b.add_input("enc2_ln1_bias", &[HIDDEN_DIM]);
    let normed2a = b.add_layer_norm(enc1_out, ln2a_eps, 1, ln2a_w, ln2a_b, &shape);

    let q2_w = b.add_input("enc2_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k2_w = b.add_input("enc2_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v2_w = b.add_input("enc2_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out2_w = b.add_input("enc2_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q2 = b.add_linear(normed2a, q2_w, None, &shape);
    let k2 = b.add_linear(normed2a, k2_w, None, &shape);
    let v2 = b.add_linear(normed2a, v2_w, None, &shape);
    let attn2 = b.add_attention(q2, k2, v2, AttentionMask::Standard, Some(scale), &shape);
    let attn2_out = b.add_linear(attn2, out2_w, None, &shape);
    let res2a = b.add_binary_add(enc1_out, attn2_out, &shape);

    let ln2b_eps = b.add_input("enc2_ln2_eps", &[1]);
    let ln2b_w = b.add_input("enc2_ln2_weight", &[HIDDEN_DIM]);
    let ln2b_b = b.add_input("enc2_ln2_bias", &[HIDDEN_DIM]);
    let normed2b = b.add_layer_norm(res2a, ln2b_eps, 1, ln2b_w, ln2b_b, &shape);

    let fc2a_w = b.add_input("enc2_fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let fc2b_w = b.add_input("enc2_fc2_weight", &[HIDDEN_DIM, FFN_DIM]);
    let h2 = b.add_linear(normed2b, fc2a_w, None, &ffn_shape);
    let h2 = b.add_gelu(h2, &ffn_shape);
    let ffn2_out = b.add_linear(h2, fc2b_w, None, &shape);
    let out = b.add_binary_add(res2a, ffn2_out, &shape);

    b.build(out)
        .expect("valid text token encoder pipeline kernel")
}

/// Bindings for text token encoder pipeline.
fn text_token_encoder_pipeline_bindings() -> Vec<TensorParamBinding> {
    let embed_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let pos_embed = ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                  // token_ids
        TensorParamBinding::ConstantTensor(embed_w),   // embed_weight
        TensorParamBinding::ConstantTensor(pos_embed), // pos_embed
        // Encoder block 1
        TensorParamBinding::ConstantScalar(1e-5), // enc1_ln1_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // enc1_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // enc1_ln1_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc1_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc1_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc1_v_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc1_out_weight
        TensorParamBinding::ConstantScalar(1e-5), // enc1_ln2_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // enc1_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // enc1_ln2_bias
        TensorParamBinding::ConstantTensor(fc1_w.clone()), // enc1_fc1_weight
        TensorParamBinding::ConstantTensor(fc2_w.clone()), // enc1_fc2_weight
        // Encoder block 2
        TensorParamBinding::ConstantScalar(1e-5), // enc2_ln1_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // enc2_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // enc2_ln1_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc2_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc2_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // enc2_v_weight
        TensorParamBinding::ConstantTensor(attn_w), // enc2_out_weight
        TensorParamBinding::ConstantScalar(1e-5), // enc2_ln2_eps
        TensorParamBinding::ConstantTensor(ln_w), // enc2_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b), // enc2_ln2_bias
        TensorParamBinding::ConstantTensor(fc1_w), // enc2_fc1_weight
        TensorParamBinding::ConstantTensor(fc2_w), // enc2_fc2_weight
    ]
}

/// IBP through text token encoder pipeline.
///
/// Embedding -> positional encoding -> 2 transformer encoder blocks.
/// Verifies bounds propagation through the full text encoding pathway.
#[test]
fn test_compose_granite_docling_text_token_encoder_ibp() {
    let def = build_text_token_encoder_pipeline_kernel();
    let bindings = text_token_encoder_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), (VOCAB_SIZE - 1) as f32),
    )
    .expect("valid token bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through text token encoder pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "text token encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Text token encoder pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through text token encoder pipeline.
///
/// Tests CROWN linearization through embedding + positional add + 2
/// transformer encoder blocks with LayerNorm, attention, and GELU MLP.
#[test]
fn test_compose_granite_docling_text_token_encoder_crown() {
    let def = build_text_token_encoder_pipeline_kernel();
    let bindings = text_token_encoder_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[SEQ_LEN]), (VOCAB_SIZE - 1) as f32),
    )
    .expect("valid token bounds");

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Text token encoder pipeline: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 42. Encoder self-attention mechanism
// ===========================================================================

/// Build an isolated encoder self-attention mechanism:
/// Q/K/V projection -> scaled dot-product attention -> output projection.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder hidden states [-1, 1]).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Tests the core self-attention block in isolation without normalization
/// or residual connections, focusing on bounds through the attention
/// computation (Q@K^T -> softmax -> @V) and linear projections.
fn build_encoder_self_attn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("encoder_self_attn");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();

    // Q/K/V projections
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let q_b = b.add_input("q_bias", &[HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_b = b.add_input("k_bias", &[HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_b = b.add_input("v_bias", &[HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, Some(q_b), &shape);
    let k = b.add_linear(input, k_w, Some(k_b), &shape);
    let v = b.add_linear(input, v_w, Some(v_b), &shape);

    // Scaled dot-product attention
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);

    // Output projection
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_b = b.add_input("out_bias", &[HIDDEN_DIM]);
    let out = b.add_linear(attn, out_w, Some(out_b), &shape);

    b.build(out).expect("valid encoder self-attention kernel")
}

/// Bindings for encoder self-attention.
fn encoder_self_attn_bindings() -> Vec<TensorParamBinding> {
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let attn_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantTensor(attn_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(attn_b.clone()), // q_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(attn_b.clone()), // k_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(attn_b.clone()), // v_bias
        TensorParamBinding::ConstantTensor(attn_w),         // out_weight
        TensorParamBinding::ConstantTensor(attn_b),         // out_bias
    ]
}

/// IBP through encoder self-attention mechanism.
///
/// Q/K/V projection -> scaled dot-product attention -> output projection.
/// Tests bounds through the core attention computation in isolation.
#[test]
fn test_compose_granite_docling_encoder_self_attn_ibp() {
    let def = build_encoder_self_attn_kernel();
    let bindings = encoder_self_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder self-attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "encoder self-attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Encoder self-attention IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through encoder self-attention mechanism.
///
/// Tests CROWN linearization through Q/K/V linear projections + softmax
/// attention + output linear projection.
#[test]
fn test_compose_granite_docling_encoder_self_attn_crown() {
    let def = build_encoder_self_attn_kernel();
    let bindings = encoder_self_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Encoder self-attention: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 43. Decoder cross-attention with residual
// ===========================================================================

/// Build decoder cross-attention: decoder queries attend to encoder key/values
/// with multi-head cross-attention and residual connection.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, decoder hidden states [-1, 1]).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// RMSNorm -> Q from decoder, K/V from frozen encoder features ->
/// scaled dot-product cross-attention -> output projection -> residual.
/// This is the cross-attention sub-layer in a Granite decoder block.
fn build_decoder_cross_attn_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("decoder_cross_attn_residual");

    let input = b.add_input("decoder_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    // Pre-cross-attention RMSNorm
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, norm_eps, 1, norm_w, &shape);

    // Q projection from decoder
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let q = b.add_linear(normed, q_w, None, &shape);

    // K/V from frozen encoder features (different sequence length = NUM_PATCHES)
    let enc_k = b.add_input("encoder_k", &[NUM_PATCHES, HIDDEN_DIM]);
    let enc_v = b.add_input("encoder_v", &[NUM_PATCHES, HIDDEN_DIM]);

    // K/V projections applied to encoder features
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k = b.add_linear(enc_k, k_w, None, &[NUM_PATCHES, HIDDEN_DIM]);
    let v = b.add_linear(enc_v, v_w, None, &[NUM_PATCHES, HIDDEN_DIM]);

    // Cross-attention: decoder queries attend to encoder key/values
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);

    // Output projection + residual
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(input, attn_out, &shape);

    b.build(out)
        .expect("valid decoder cross-attention with residual kernel")
}

/// Bindings for decoder cross-attention with residual.
fn decoder_cross_attn_residual_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let enc_feat = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                         // decoder_hidden
        TensorParamBinding::ConstantScalar(1e-5),             // norm_eps
        TensorParamBinding::ConstantTensor(norm_w),           // norm_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()),   // q_weight
        TensorParamBinding::ConstantTensor(enc_feat.clone()), // encoder_k
        TensorParamBinding::ConstantTensor(enc_feat),         // encoder_v
        TensorParamBinding::ConstantTensor(attn_w.clone()),   // k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()),   // v_weight
        TensorParamBinding::ConstantTensor(attn_w),           // out_weight
    ]
}

/// IBP through decoder cross-attention with residual.
///
/// Decoder queries attend to encoder key/values via multi-head cross-attention
/// with RMSNorm pre-normalization and residual connection.
#[test]
fn test_compose_granite_docling_decoder_cross_attn_residual_ibp() {
    let def = build_decoder_cross_attn_residual_kernel();
    let bindings = decoder_cross_attn_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder cross-attention with residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "decoder cross-attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Decoder cross-attention residual IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through decoder cross-attention with residual.
///
/// Tests CROWN linearization through RMSNorm -> Q linear -> cross-attention
/// (softmax over encoder keys) -> output linear -> residual add.
#[test]
fn test_compose_granite_docling_decoder_cross_attn_residual_crown() {
    let def = build_decoder_cross_attn_residual_kernel();
    let bindings = decoder_cross_attn_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Decoder cross-attention residual: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 44. Output projection and classification head
// ===========================================================================

/// Build output projection and classification head:
/// decoder output -> LayerNorm -> linear projection -> softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, decoder output states [-1, 1]).
/// Output: `[SEQ_LEN, NUM_CLASSES]` (softmax class probabilities).
///
/// Tests the final stage of an encoder-decoder model where decoder
/// hidden states are normalized, projected to class logits, and converted
/// to probability distributions via softmax.
fn build_output_proj_classification_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("output_proj_classification_head");

    let input = b.add_input("decoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Layer normalization
    let ln_eps = b.add_input("ln_eps", &[1]);
    let ln_w = b.add_input("ln_weight", &[HIDDEN_DIM]);
    let ln_b = b.add_input("ln_bias", &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &shape);

    // Linear projection to class logits
    let proj_w = b.add_input("proj_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(normed, proj_w, Some(proj_b), &[SEQ_LEN, NUM_CLASSES]);

    // Softmax over classes (axis 1)
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_CLASSES]);

    b.build(out)
        .expect("valid output projection + classification head kernel")
}

/// Bindings for output projection and classification head.
fn output_proj_classification_head_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, HIDDEN_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);

    vec![
        TensorParamBinding::Variable,               // decoder_output
        TensorParamBinding::ConstantScalar(1e-5),   // ln_eps
        TensorParamBinding::ConstantTensor(ln_w),   // ln_weight
        TensorParamBinding::ConstantTensor(ln_b),   // ln_bias
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
        TensorParamBinding::ConstantTensor(proj_b), // proj_bias
    ]
}

/// IBP through output projection and classification head.
///
/// LayerNorm -> linear -> softmax. Softmax output must be in [0, 1].
#[test]
fn test_compose_granite_docling_output_proj_cls_head_ibp() {
    let def = build_output_proj_classification_head_kernel();
    let bindings = output_proj_classification_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through output projection + classification head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, NUM_CLASSES],
        "classification head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Output projection + cls head IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-6,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "softmax upper bound should be <= 1, got {hi_max}"
    );
}

/// CROWN through output projection and classification head.
///
/// Tests CROWN linearization through LayerNorm -> linear -> softmax.
#[test]
fn test_compose_granite_docling_output_proj_cls_head_crown() {
    let def = build_output_proj_classification_head_kernel();
    let bindings = output_proj_classification_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Output proj + cls head: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-6,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "softmax upper bound should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 45. Full end-to-end pipeline: image encoder + text encoder -> decoder
//     with cross-attention -> output classification head
// ===========================================================================

/// Build a full end-to-end encoder-decoder pipeline:
/// Image encoder (patch embed + 1 ViT block) + text encoder (embedding +
/// 1 transformer block) -> decoder with cross-attention -> classification head.
///
/// Image input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[NUM_PATCHES, NUM_CLASSES]` (softmax class probabilities).
///
/// This is the most comprehensive pipeline test: vision features are encoded
/// via patch embedding + transformer, then a Granite decoder block attends to
/// those features via cross-attention, and a classification head produces
/// probability distributions. Tests bounds composition across the full
/// encoder-decoder architecture.
fn build_full_enc_dec_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("full_enc_dec_pipeline");

    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];
    let enc_ffn_shape = [NUM_PATCHES, FFN_DIM];
    let dec_shape = [NUM_PATCHES, HIDDEN_DIM];
    let dec_ffn_shape = [NUM_PATCHES, FFN_DIM];
    let scale_enc = 1.0 / (SIGLIP2_HEAD_DIM as f32).sqrt();
    let scale_dec = 1.0 / (GRANITE_HEAD_DIM as f32).sqrt();

    // ===== Image encoder =====

    // Patch embedding
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_b = b.add_input("patch_bias", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        image,
        patch_w,
        Some(patch_b),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &patch_shape);

    // Positional encoding
    let img_pos = b.add_input("img_pos_embed", &[NUM_PATCHES, HIDDEN_DIM]);
    let img_positioned = b.add_binary_add(patches, img_pos, &patch_shape);

    // Vision encoder block (LayerNorm -> attention -> residual -> LN -> GELU MLP -> residual)
    let vln1_eps = b.add_input("venc_ln1_eps", &[1]);
    let vln1_w = b.add_input("venc_ln1_weight", &[HIDDEN_DIM]);
    let vln1_b = b.add_input("venc_ln1_bias", &[HIDDEN_DIM]);
    let vnormed1 = b.add_layer_norm(img_positioned, vln1_eps, 1, vln1_w, vln1_b, &patch_shape);

    let vq_w = b.add_input("venc_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vk_w = b.add_input("venc_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vv_w = b.add_input("venc_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vout_w = b.add_input("venc_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let vq = b.add_linear(vnormed1, vq_w, None, &patch_shape);
    let vk = b.add_linear(vnormed1, vk_w, None, &patch_shape);
    let vv = b.add_linear(vnormed1, vv_w, None, &patch_shape);
    let vattn = b.add_attention(
        vq,
        vk,
        vv,
        AttentionMask::Standard,
        Some(scale_enc),
        &patch_shape,
    );
    let vattn_out = b.add_linear(vattn, vout_w, None, &patch_shape);
    let vres1 = b.add_binary_add(img_positioned, vattn_out, &patch_shape);

    let vln2_eps = b.add_input("venc_ln2_eps", &[1]);
    let vln2_w = b.add_input("venc_ln2_weight", &[HIDDEN_DIM]);
    let vln2_b = b.add_input("venc_ln2_bias", &[HIDDEN_DIM]);
    let vnormed2 = b.add_layer_norm(vres1, vln2_eps, 1, vln2_w, vln2_b, &patch_shape);

    let vfc1_w = b.add_input("venc_fc1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let vfc2_w = b.add_input("venc_fc2_weight", &[HIDDEN_DIM, FFN_DIM]);
    let vh = b.add_linear(vnormed2, vfc1_w, None, &enc_ffn_shape);
    let vh = b.add_gelu(vh, &enc_ffn_shape);
    let vffn_out = b.add_linear(vh, vfc2_w, None, &patch_shape);
    let vision_features = b.add_binary_add(vres1, vffn_out, &patch_shape);

    // Vision projection to decoder space
    let vproj_w = b.add_input("vproj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vproj_b = b.add_input("vproj_bias", &[HIDDEN_DIM]);
    let vision_projected = b.add_linear(vision_features, vproj_w, Some(vproj_b), &patch_shape);

    // ===== Decoder with cross-attention =====

    // Use vision_projected as initial decoder state (treating patches as decoder sequence)
    let dec_input = vision_projected;

    // Decoder self-attention
    let dec_norm1_eps = b.add_input("dec_norm1_eps", &[1]);
    let dec_norm1_w = b.add_input("dec_norm1_weight", &[HIDDEN_DIM]);
    let dec_normed1 = b.add_rms_norm(dec_input, dec_norm1_eps, 1, dec_norm1_w, &dec_shape);

    let dec_q_w = b.add_input("dec_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let dec_k_w = b.add_input("dec_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let dec_v_w = b.add_input("dec_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let dec_out_w = b.add_input("dec_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let dec_q = b.add_linear(dec_normed1, dec_q_w, None, &dec_shape);
    let dec_k = b.add_linear(dec_normed1, dec_k_w, None, &dec_shape);
    let dec_v = b.add_linear(dec_normed1, dec_v_w, None, &dec_shape);
    let dec_attn = b.add_attention(
        dec_q,
        dec_k,
        dec_v,
        AttentionMask::Causal,
        Some(scale_dec),
        &dec_shape,
    );
    let dec_attn_out = b.add_linear(dec_attn, dec_out_w, None, &dec_shape);
    let dec_res1 = b.add_binary_add(dec_input, dec_attn_out, &dec_shape);

    // Decoder SwiGLU FFN
    let dec_norm2_eps = b.add_input("dec_norm2_eps", &[1]);
    let dec_norm2_w = b.add_input("dec_norm2_weight", &[HIDDEN_DIM]);
    let dec_normed2 = b.add_rms_norm(dec_res1, dec_norm2_eps, 1, dec_norm2_w, &dec_shape);

    let dec_gate_w = b.add_input("dec_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let dec_up_w = b.add_input("dec_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let dec_down_w = b.add_input("dec_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let dec_gate = b.add_linear(dec_normed2, dec_gate_w, None, &dec_ffn_shape);
    let dec_gate_sig = b.add_sigmoid(dec_gate, &dec_ffn_shape);
    let dec_gate_act = b.add_binary_mul(dec_gate, dec_gate_sig, &dec_ffn_shape);
    let dec_up = b.add_linear(dec_normed2, dec_up_w, None, &dec_ffn_shape);
    let dec_hidden = b.add_binary_mul(dec_gate_act, dec_up, &dec_ffn_shape);
    let dec_ffn_out = b.add_linear(dec_hidden, dec_down_w, None, &dec_shape);
    let dec_out = b.add_binary_add(dec_res1, dec_ffn_out, &dec_shape);

    // ===== Classification head =====

    // Final layer norm
    let final_ln_eps = b.add_input("final_ln_eps", &[1]);
    let final_ln_w = b.add_input("final_ln_weight", &[HIDDEN_DIM]);
    let final_ln_b = b.add_input("final_ln_bias", &[HIDDEN_DIM]);
    let final_normed =
        b.add_layer_norm(dec_out, final_ln_eps, 1, final_ln_w, final_ln_b, &dec_shape);

    // Linear classification head + softmax
    let cls_w = b.add_input("cls_weight", &[NUM_CLASSES, HIDDEN_DIM]);
    let cls_b = b.add_input("cls_bias", &[NUM_CLASSES]);
    let logits = b.add_linear(
        final_normed,
        cls_w,
        Some(cls_b),
        &[NUM_PATCHES, NUM_CLASSES],
    );
    let out = b.add_softmax(logits, 1, &[NUM_PATCHES, NUM_CLASSES]);

    b.build(out)
        .expect("valid full encoder-decoder pipeline kernel")
}

/// Bindings for full end-to-end encoder-decoder pipeline.
fn full_enc_dec_pipeline_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fc2_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let pos_embed = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);
    let cls_w = ArrayD::from_elem(IxDyn(&[NUM_CLASSES, HIDDEN_DIM]), WEIGHT_MAG);
    let cls_b = ArrayD::from_elem(IxDyn(&[NUM_CLASSES]), 0.0f32);

    vec![
        TensorParamBinding::Variable, // image
        // Patch embedding
        TensorParamBinding::ConstantTensor(patch_w), // patch_weight
        TensorParamBinding::ConstantTensor(patch_b), // patch_bias
        TensorParamBinding::ConstantTensor(pos_embed), // img_pos_embed
        // Vision encoder block
        TensorParamBinding::ConstantScalar(1e-5), // venc_ln1_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // venc_ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // venc_ln1_bias
        TensorParamBinding::ConstantTensor(attn_w.clone()), // venc_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // venc_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // venc_v_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // venc_out_weight
        TensorParamBinding::ConstantScalar(1e-5), // venc_ln2_eps
        TensorParamBinding::ConstantTensor(ln_w.clone()), // venc_ln2_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // venc_ln2_bias
        TensorParamBinding::ConstantTensor(fc1_w.clone()), // venc_fc1_weight
        TensorParamBinding::ConstantTensor(fc2_w.clone()), // venc_fc2_weight
        // Vision projection
        TensorParamBinding::ConstantTensor(attn_w.clone()), // vproj_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)), // vproj_bias
        // Decoder self-attention
        TensorParamBinding::ConstantScalar(1e-5), // dec_norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // dec_norm1_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // dec_q_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // dec_k_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()), // dec_v_weight
        TensorParamBinding::ConstantTensor(attn_w), // dec_out_weight
        // Decoder SwiGLU FFN
        TensorParamBinding::ConstantScalar(1e-5), // dec_norm2_eps
        TensorParamBinding::ConstantTensor(norm_w), // dec_norm2_weight
        TensorParamBinding::ConstantTensor(fc1_w), // dec_gate_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // dec_up_weight
        TensorParamBinding::ConstantTensor(fc2_w), // dec_down_weight
        // Classification head
        TensorParamBinding::ConstantScalar(1e-5), // final_ln_eps
        TensorParamBinding::ConstantTensor(ln_w), // final_ln_weight
        TensorParamBinding::ConstantTensor(ln_b), // final_ln_bias
        TensorParamBinding::ConstantTensor(cls_w), // cls_weight
        TensorParamBinding::ConstantTensor(cls_b), // cls_bias
    ]
}

/// IBP through full end-to-end encoder-decoder pipeline.
///
/// Image encoder (patch embed + ViT block + projection) -> Granite decoder
/// (self-attention + SwiGLU FFN) -> LayerNorm -> classification head -> softmax.
/// Verifies bounds compose across the entire encoder-decoder architecture.
/// Softmax output must be in [0, 1].
#[test]
fn test_compose_granite_docling_full_enc_dec_pipeline_ibp() {
    let def = build_full_enc_dec_pipeline_kernel();
    let bindings = full_enc_dec_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full encoder-decoder pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, NUM_CLASSES],
        "full enc-dec pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full enc-dec pipeline IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-6,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "softmax upper bound should be <= 1, got {hi_max}"
    );
}

/// CROWN through full end-to-end encoder-decoder pipeline.
///
/// Deepest end-to-end pipeline test: image -> patch embed -> ViT encoder ->
/// projection -> Granite decoder (RMSNorm + causal attention + SwiGLU) ->
/// LayerNorm -> classification softmax.
/// Tests CROWN linearization depth through all major layer types.
#[test]
fn test_compose_granite_docling_full_enc_dec_pipeline_crown() {
    let def = build_full_enc_dec_pipeline_kernel();
    let bindings = full_enc_dec_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_PATCHES, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full enc-dec pipeline: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-6,
        "softmax lower bound should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-6,
        "softmax upper bound should be <= 1, got {hi_max}"
    );
}
