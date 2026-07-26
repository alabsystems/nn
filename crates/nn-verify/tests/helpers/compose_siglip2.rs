// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: SigLIP2 vision encoder NY composition.
//!
//! Verifies bounds propagation through SigLIP2-specific sub-blocks:
//!
//! 1. **Patch embedding**: Conv2d(3, D, P, stride=P) -> reshape -> transpose
//! 2. **Position embedding addition**: patch_emb + pos_emb (constant + variable)
//! 3. **SiGLU FFN block**: Linear -> SiLU gate + Linear up -> mul -> Linear down
//! 4. **Full transformer block**: LayerNorm -> MHA -> residual -> LayerNorm -> SiGLU FFN -> residual
//! 5. **verify_and_assert** recording for each sub-block
//!
//! SigLIP2 architecture (Zhai et al. 2023):
//! - Patch embedding: Conv2d with kernel_size = stride = patch_size
//! - SiGLU FFN: gate branch uses SiLU instead of GELU, multiplicative gating
//!   output = fc_down(silu(fc_gate(x)) * fc_up(x))
//! - Standard bidirectional self-attention (no causal masking)
//! - LayerNorm (pre-norm transformer)
//!
//! Dimensions (small for fast verification):
//! - IMG_SIZE=32, PATCH_SIZE=16, EMBED_DIM=64, NUM_HEADS=4, FFN_DIM=128
//! - NUM_PATCHES = (32/16)^2 = 4, SEQ_LEN = 4 + 1 = 5 (with CLS token)
//!
//! Part of #3540: SigLIP2 end-to-end NY compose bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{
    tensor_kernel_to_graph, BoundedTensor, TensorParamBinding, VerificationSoundnessMode,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions — small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Image height and width (square image).
const IMG_SIZE: usize = 32;
/// Patch size (P). IMG_SIZE must be divisible by PATCH_SIZE.
const PATCH_SIZE: usize = 16;
/// Number of patches per spatial dimension.
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 2
/// Total number of patches.
const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE; // 4
/// Sequence length including CLS token: NUM_PATCHES + 1.
const SEQ_LEN: usize = NUM_PATCHES + 1; // 5
/// Input channels (RGB).
const IN_CHANNELS: usize = 3;
/// Embedding dimension (tiny SigLIP2 hidden size).
const EMBED_DIM: usize = 64;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// FFN intermediate dimension (SiGLU uses separate gate and up projections).
const FFN_DIM: usize = 128;

// ---------------------------------------------------------------------------
// 1. Patch embedding: Conv2d -> reshape -> transpose
// ---------------------------------------------------------------------------

/// Build a SigLIP2 patch embedding kernel using Conv2d.
///
/// Input: `[3, IMG_SIZE, IMG_SIZE]` (Variable, image pixels in [0, 1]).
/// Output: `[NUM_PATCHES, EMBED_DIM]` after reshape and transpose.
///
/// Conv2d(in_channels=3, out_channels=D, kernel=P, stride=P, padding=0)
/// produces `[D, GRID_SIZE, GRID_SIZE]`.
/// Reshape to `[D, NUM_PATCHES]`, then transpose to `[NUM_PATCHES, D]`.
fn build_siglip2_patch_embedding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_patch_embedding");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let weight = b.add_input(
        "proj_weight",
        &[EMBED_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let bias = b.add_input("proj_bias", &[EMBED_DIM]);

    // Conv2d: [3, 32, 32] -> [D, 2, 2]
    let conv_out = b.add_conv2d(
        input,
        weight,
        Some(bias),
        PATCH_SIZE, // stride_h
        PATCH_SIZE, // stride_w
        0,          // padding_h
        0,          // padding_w
        &[EMBED_DIM, GRID_SIZE, GRID_SIZE],
    );

    // Reshape: [D, 2, 2] -> [D, NUM_PATCHES]
    let reshaped = b.add_reshape(conv_out, &[EMBED_DIM, NUM_PATCHES]);

    // Transpose: [D, NUM_PATCHES] -> [NUM_PATCHES, D]
    let out = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, EMBED_DIM]);

    b.build(out).expect("valid SigLIP2 patch embedding kernel")
}

/// Bindings for SigLIP2 patch embedding.
fn siglip2_patch_embedding_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(
        IxDyn(&[EMBED_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        0.02f32,
    );
    let bias = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // image [3, 32, 32]
        TensorParamBinding::ConstantTensor(w),    // proj_weight [D, 3, P, P]
        TensorParamBinding::ConstantTensor(bias), // proj_bias [D]
    ]
}

/// Create image-domain input bounds: pixels in [0, 1].
fn image_bounds_01(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

// ---------------------------------------------------------------------------
// 2. Position embedding addition
// ---------------------------------------------------------------------------

/// Build a position embedding addition kernel.
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable, from patch_emb + CLS token).
/// Output: `[SEQ_LEN, EMBED_DIM]`.
///
/// Models `output = input + pos_embed` where pos_embed is a learned constant.
fn build_siglip2_position_embedding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_position_embedding");

    let input = b.add_input("patch_tokens", &[SEQ_LEN, EMBED_DIM]);
    let pos_embed = b.add_input("pos_embed", &[SEQ_LEN, EMBED_DIM]);

    let out = b.add_binary_add(input, pos_embed, &[SEQ_LEN, EMBED_DIM]);

    b.build(out)
        .expect("valid SigLIP2 position embedding kernel")
}

/// Bindings for position embedding addition.
fn siglip2_position_embedding_bindings() -> Vec<TensorParamBinding> {
    let pos_embed = ArrayD::from_elem(IxDyn(&[SEQ_LEN, EMBED_DIM]), 0.01f32);

    vec![
        TensorParamBinding::Variable, // patch_tokens [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantTensor(pos_embed), // pos_embed [SEQ_LEN, EMBED_DIM]
    ]
}

// ---------------------------------------------------------------------------
// 3. SiGLU FFN block: Linear -> SiLU gate * Linear up -> Linear down
// ---------------------------------------------------------------------------

/// Build a SiGLU FFN kernel (SigLIP2 style).
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable).
/// Output: `[SEQ_LEN, EMBED_DIM]`.
///
/// SiGLU FFN architecture:
///   gate = fc_gate(x)                    [SEQ_LEN, FFN_DIM]
///   gate_activated = silu(gate)          [SEQ_LEN, FFN_DIM]
///   up = fc_up(x)                        [SEQ_LEN, FFN_DIM]
///   hidden = gate_activated * up         [SEQ_LEN, FFN_DIM]
///   output = fc_down(hidden)             [SEQ_LEN, EMBED_DIM]
///
/// SiLU = x * sigmoid(x), decomposed as sigmoid + binary_mul since
/// TensorBlockBuilder has no native add_silu.
fn build_siglip2_siglu_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_siglu_ffn");

    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, EMBED_DIM]);
    let gate_b = b.add_input("gate_bias", &[FFN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, EMBED_DIM]);
    let up_b = b.add_input("up_bias", &[FFN_DIM]);
    let down_w = b.add_input("down_weight", &[EMBED_DIM, FFN_DIM]);
    let down_b = b.add_input("down_bias", &[EMBED_DIM]);

    // Gate branch: Linear -> SiLU
    // fc_gate: [S, D] -> [S, FFN]
    let gate = b.add_linear(input, gate_w, Some(gate_b), &[SEQ_LEN, FFN_DIM]);
    // SiLU(x) = x * sigmoid(x): decomposed into sigmoid + binary_mul
    let gate_sig = b.add_sigmoid(gate, &[SEQ_LEN, FFN_DIM]);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &[SEQ_LEN, FFN_DIM]);

    // Up branch: Linear
    // fc_up: [S, D] -> [S, FFN]
    let up = b.add_linear(input, up_w, Some(up_b), &[SEQ_LEN, FFN_DIM]);

    // Multiplicative gating: gate_activated * up
    let hidden = b.add_binary_mul(gate_activated, up, &[SEQ_LEN, FFN_DIM]);

    // Down projection: [S, FFN] -> [S, D]
    let out = b.add_linear(hidden, down_w, Some(down_b), &[SEQ_LEN, EMBED_DIM]);

    b.build(out).expect("valid SigLIP2 SiGLU FFN kernel")
}

/// Bindings for SiGLU FFN.
fn siglip2_siglu_ffn_bindings() -> Vec<TensorParamBinding> {
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let gate_b = ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let up_b = ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32);
    let down_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM, FFN_DIM]), 0.02f32);
    let down_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,               // x [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantTensor(gate_w), // gate_weight [FFN_DIM, EMBED_DIM]
        TensorParamBinding::ConstantTensor(gate_b), // gate_bias [FFN_DIM]
        TensorParamBinding::ConstantTensor(up_w),   // up_weight [FFN_DIM, EMBED_DIM]
        TensorParamBinding::ConstantTensor(up_b),   // up_bias [FFN_DIM]
        TensorParamBinding::ConstantTensor(down_w), // down_weight [EMBED_DIM, FFN_DIM]
        TensorParamBinding::ConstantTensor(down_b), // down_bias [EMBED_DIM]
    ]
}

// ---------------------------------------------------------------------------
// 4. Full transformer block with SiGLU FFN
// ---------------------------------------------------------------------------

/// Build a full SigLIP2 transformer block with SiGLU FFN.
///
/// Input: `[SEQ_LEN, EMBED_DIM]` (Variable).
/// Output: `[SEQ_LEN, EMBED_DIM]`.
///
/// Architecture (pre-norm):
///   x1 = x + MHA(LayerNorm(x))
///   x2 = x1 + SiGLU_FFN(LayerNorm(x1))
///
/// Constructed manually (not using `add_transformer_block`) because the standard
/// builder uses GELU FFN. SigLIP2 uses SiGLU instead.
fn build_siglip2_transformer_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("siglip2_transformer_block");

    let input = b.add_input("x", &[SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    // Attention sub-block weights
    let ln1_w = b.add_input("ln1_weight", &[EMBED_DIM]);
    let ln1_b = b.add_input("ln1_bias", &[EMBED_DIM]);
    let q_w = b.add_input("q_weight", &[EMBED_DIM, EMBED_DIM]);
    let k_w = b.add_input("k_weight", &[EMBED_DIM, EMBED_DIM]);
    let v_w = b.add_input("v_weight", &[EMBED_DIM, EMBED_DIM]);
    let out_w = b.add_input("out_weight", &[EMBED_DIM, EMBED_DIM]);

    // SiGLU FFN sub-block weights
    let ln2_w = b.add_input("ln2_weight", &[EMBED_DIM]);
    let ln2_b = b.add_input("ln2_bias", &[EMBED_DIM]);
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, EMBED_DIM]);
    let gate_b = b.add_input("gate_bias", &[FFN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, EMBED_DIM]);
    let up_b = b.add_input("up_bias", &[FFN_DIM]);
    let down_w = b.add_input("down_weight", &[EMBED_DIM, FFN_DIM]);
    let down_b = b.add_input("down_bias", &[EMBED_DIM]);

    // --- Attention sub-block: LayerNorm -> MHA -> residual ---

    // LayerNorm1: [S, D] -> [S, D]
    let normed1 = b.add_layer_norm(input, eps, 1, ln1_w, ln1_b, &[SEQ_LEN, EMBED_DIM]);

    // MHA composite: internally does Linear(Q,K,V) -> Reshape -> Transpose ->
    // Attention -> Transpose -> Reshape -> Linear(out).
    // Uses bidirectional attention (Standard mask) for vision encoder.
    let attn_out = b
        .add_multi_head_attention(
            normed1,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, EMBED_DIM],
        )
        .expect("valid MHA");

    // Residual 1: x + attn_out
    let x1 = b.add_binary_add(input, attn_out, &[SEQ_LEN, EMBED_DIM]);

    // --- SiGLU FFN sub-block: LayerNorm -> SiGLU -> residual ---

    // LayerNorm2: [S, D] -> [S, D]
    let normed2 = b.add_layer_norm(x1, eps, 1, ln2_w, ln2_b, &[SEQ_LEN, EMBED_DIM]);

    // SiGLU gate branch: Linear -> SiLU (= x * sigmoid(x))
    let gate = b.add_linear(normed2, gate_w, Some(gate_b), &[SEQ_LEN, FFN_DIM]);
    let gate_sig = b.add_sigmoid(gate, &[SEQ_LEN, FFN_DIM]);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &[SEQ_LEN, FFN_DIM]);

    // SiGLU up branch: Linear
    let up = b.add_linear(normed2, up_w, Some(up_b), &[SEQ_LEN, FFN_DIM]);

    // Multiplicative gating: silu(gate(x)) * up(x)
    let hidden = b.add_binary_mul(gate_activated, up, &[SEQ_LEN, FFN_DIM]);

    // Down projection: [S, FFN] -> [S, D]
    let ffn_out = b.add_linear(hidden, down_w, Some(down_b), &[SEQ_LEN, EMBED_DIM]);

    // Residual 2: x1 + ffn_out
    let x2 = b.add_binary_add(x1, ffn_out, &[SEQ_LEN, EMBED_DIM]);

    b.build(x2).expect("valid SigLIP2 transformer block kernel")
}

/// Bindings for the full SigLIP2 transformer block.
fn siglip2_transformer_block_bindings() -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[EMBED_DIM, EMBED_DIM]), 0.02f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let gate_b = ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, EMBED_DIM]), 0.02f32);
    let up_b = ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32);
    let down_w = ArrayD::from_elem(IxDyn(&[EMBED_DIM, FFN_DIM]), 0.02f32);
    let down_b = ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // x [SEQ_LEN, EMBED_DIM]
        TensorParamBinding::ConstantScalar(1e-5), // eps [1]
        // Attention sub-block
        TensorParamBinding::ConstantTensor(ln_w.clone()), // ln1_weight
        TensorParamBinding::ConstantTensor(ln_b.clone()), // ln1_bias
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight
        TensorParamBinding::ConstantTensor(w_proj),       // out_weight
        // SiGLU FFN sub-block
        TensorParamBinding::ConstantTensor(ln_w), // ln2_weight
        TensorParamBinding::ConstantTensor(ln_b), // ln2_bias
        TensorParamBinding::ConstantTensor(gate_w), // gate_weight
        TensorParamBinding::ConstantTensor(gate_b), // gate_bias
        TensorParamBinding::ConstantTensor(up_w), // up_weight
        TensorParamBinding::ConstantTensor(up_b), // up_bias
        TensorParamBinding::ConstantTensor(down_w), // down_weight
        TensorParamBinding::ConstantTensor(down_b), // down_bias
    ]
}

// ===========================================================================
// Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. Patch embedding tests
// ---------------------------------------------------------------------------

/// SigLIP2 patch embedding TensorKernelDef validates.
#[test]
fn test_siglip2_patch_embedding_def_validates() {
    let def = build_siglip2_patch_embedding_kernel();
    def.validate()
        .expect("SigLIP2 patch embedding kernel should validate");
}

/// Patch embedding translates to NY GraphNetwork.
#[test]
fn test_siglip2_patch_embedding_graph_builds() {
    let def = build_siglip2_patch_embedding_kernel();
    let bindings = siglip2_patch_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("SigLIP2 patch embedding graph should translate");

    // Conv2d + Reshape + Transpose = at least 3 nodes.
    assert!(
        graph.num_nodes() >= 3,
        "patch embedding graph should have >= 3 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through SigLIP2 patch embedding with [0, 1] image input.
#[test]
fn test_siglip2_patch_embedding_ibp_propagates() {
    let def = build_siglip2_patch_embedding_kernel();
    let bindings = siglip2_patch_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through SigLIP2 patch embedding");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, EMBED_DIM],
        "output shape should be [NUM_PATCHES={NUM_PATCHES}, EMBED_DIM={EMBED_DIM}], got {:?}",
        lo.shape()
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 patch embedding IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

/// CROWN bounds propagate through SigLIP2 patch embedding.
#[test]
fn test_siglip2_patch_embedding_crown_propagation() {
    let def = build_siglip2_patch_embedding_kernel();
    let bindings = siglip2_patch_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 patch embedding: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// Patch embedding verify and record under "siglip2_patch_embedding" key.
#[test]
fn test_siglip2_patch_embedding_verify_and_record() {
    let def = build_siglip2_patch_embedding_kernel();
    let bindings = siglip2_patch_embedding_bindings();
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let result = verify_and_assert(&def, &bindings, &input, "siglip2_patch_embedding");
    assert_eq!(result.num_variables, 1, "single Variable input (image)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, EMBED_DIM]);
}

// ---------------------------------------------------------------------------
// 2. Position embedding tests
// ---------------------------------------------------------------------------

/// Position embedding addition TensorKernelDef validates.
#[test]
fn test_siglip2_position_embedding_def_validates() {
    let def = build_siglip2_position_embedding_kernel();
    def.validate()
        .expect("SigLIP2 position embedding kernel should validate");
}

/// Position embedding graph translates.
#[test]
fn test_siglip2_position_embedding_graph_builds() {
    let def = build_siglip2_position_embedding_kernel();
    let bindings = siglip2_position_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("SigLIP2 position embedding graph should translate");

    assert!(
        graph.num_nodes() >= 1,
        "position embedding graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through position embedding addition.
///
/// Input tokens in [-1, 1], pos_embed is a constant. Output should be
/// shifted by the constant offset.
#[test]
fn test_siglip2_position_embedding_ibp_propagates() {
    let def = build_siglip2_position_embedding_kernel();
    let bindings = siglip2_position_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through position embedding");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 position embedding IBP: bounds=[{lo_min}, {hi_max}]");

    // With [-1, 1] input + 0.01 pos_embed: output in [-0.99, 1.01]
    assert!(lo_min >= -1.5, "IBP lower should be >= -1.5, got {lo_min}");
    assert!(hi_max <= 1.5, "IBP upper should be <= 1.5, got {hi_max}");
}

/// CROWN bounds through position embedding.
#[test]
fn test_siglip2_position_embedding_crown_propagation() {
    let def = build_siglip2_position_embedding_kernel();
    let bindings = siglip2_position_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 position embedding: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Position embedding verify and record.
#[test]
fn test_siglip2_position_embedding_verify_and_record() {
    let def = build_siglip2_position_embedding_kernel();
    let bindings = siglip2_position_embedding_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "siglip2_position_embedding");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, EMBED_DIM]);
}

// ---------------------------------------------------------------------------
// 3. SiGLU FFN tests
// ---------------------------------------------------------------------------

/// SiGLU FFN TensorKernelDef validates.
#[test]
fn test_siglip2_siglu_ffn_def_validates() {
    let def = build_siglip2_siglu_ffn_kernel();
    def.validate()
        .expect("SigLIP2 SiGLU FFN kernel should validate");
}

/// SiGLU FFN translates to NY GraphNetwork.
#[test]
fn test_siglip2_siglu_ffn_graph_builds() {
    let def = build_siglip2_siglu_ffn_kernel();
    let bindings = siglip2_siglu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("SiGLU FFN graph should translate");

    // 3 Linear + Sigmoid + 2 BinaryMul = at least 6 nodes.
    assert!(
        graph.num_nodes() >= 6,
        "SiGLU FFN graph should have >= 6 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through SiGLU FFN.
///
/// SiGLU uses multiplicative gating: silu(gate(x)) * up(x). The sigmoid
/// in SiLU bounds gate_activated to [0, max], and the multiplication
/// produces bounded output. With small weights, output should be small.
#[test]
fn test_siglip2_siglu_ffn_ibp_propagates() {
    let def = build_siglip2_siglu_ffn_kernel();
    let bindings = siglip2_siglu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through SiGLU FFN");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 SiGLU FFN IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

/// CROWN bounds propagate through SiGLU FFN.
///
/// Sigmoid in SiLU is piecewise-smooth and can be linearized by CROWN.
/// The multiplicative gating (BinaryMul) may cause CROWN to fall back
/// to IBP in some configurations.
#[test]
fn test_siglip2_siglu_ffn_crown_propagation() {
    let def = build_siglip2_siglu_ffn_kernel();
    let bindings = siglip2_siglu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 SiGLU FFN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "output lower bound must be finite");
    assert!(hi_max.is_finite(), "output upper bound must be finite");
}

/// SiGLU FFN verify and record.
#[test]
fn test_siglip2_siglu_ffn_verify_and_record() {
    let def = build_siglip2_siglu_ffn_kernel();
    let bindings = siglip2_siglu_ffn_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "siglip2_siglu_ffn");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, EMBED_DIM]);
}

// ---------------------------------------------------------------------------
// 4. Full transformer block tests
// ---------------------------------------------------------------------------

/// Full SigLIP2 transformer block TensorKernelDef validates.
#[test]
fn test_siglip2_transformer_block_def_validates() {
    let def = build_siglip2_transformer_block_kernel();
    def.validate()
        .expect("SigLIP2 transformer block kernel should validate");
}

/// Full transformer block translates to NY GraphNetwork.
#[test]
fn test_siglip2_transformer_block_graph_builds() {
    let def = build_siglip2_transformer_block_kernel();
    let bindings = siglip2_transformer_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("SigLIP2 transformer block graph should translate");

    // LayerNorm + QKV proj + MHA + out proj + residual
    // + LayerNorm + 3 Linear (gate/up/down) + Sigmoid + 2 BinaryMul + residual
    // = at least 15 nodes.
    assert!(
        graph.num_nodes() >= 15,
        "transformer block graph should have >= 15 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full SigLIP2 transformer block.
#[test]
fn test_siglip2_transformer_block_ibp_propagates() {
    let def = build_siglip2_transformer_block_kernel();
    let bindings = siglip2_transformer_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through SigLIP2 transformer block");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 transformer block IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN bounds propagate through the full SigLIP2 transformer block.
///
/// LayerNorm requires heuristic linearization (IbpValidated mode).
/// Sigmoid in SiLU linearizes cleanly. The multiplicative gating in SiGLU
/// and the attention softmax may cause CROWN to produce wider bounds.
#[test]
fn test_siglip2_transformer_block_crown_propagation() {
    let def = build_siglip2_transformer_block_kernel();
    let bindings = siglip2_transformer_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, EMBED_DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SigLIP2 transformer block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Full transformer block verify and record.
///
/// LayerNorm causes heuristic normalization approximation, so soundness
/// mode should be Heuristic.
#[test]
fn test_siglip2_transformer_block_verify_and_record() {
    let def = build_siglip2_transformer_block_kernel();
    let bindings = siglip2_transformer_block_bindings();
    let input = uniform_bounds(&[SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "siglip2_transformer_block");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, EMBED_DIM]);

    // LayerNorm uses heuristic normalization approximation -> Heuristic mode.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "SigLIP2 transformer block with LayerNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
