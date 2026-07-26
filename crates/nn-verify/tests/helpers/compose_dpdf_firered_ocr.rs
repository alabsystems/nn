// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: FireRed-OCR subgraph NY composition.
//!
//! Verifies bounds propagation through FireRed-OCR sub-blocks used in the
//! dpdf document understanding pipeline for optical character recognition.
//! FireRed-OCR is a Qwen3-VL-2B variant fine-tuned for document OCR with
//! a CTC decoding head.
//!
//! 1. **Patch embedding IBP**: Conv2d(3, 1536, 14, stride=14) for 2B-scale
//!    patch tokenization. Maps image patches to hidden dimension.
//!
//! 2. **Small attention IBP**: 12-head attention at 1536 dims (smaller than
//!    30B Qwen3-VL). Self-attention over patch sequence with residual.
//!
//! 3. **Encoder layer IBP**: RMSNorm -> Attention -> residual -> SwiGLU FFN
//!    -> residual. One complete vision encoder block.
//!
//! 4. **Encoder layer CROWN**: Same encoder layer with CROWN linearization
//!    through RMSNorm for tighter bounds.
//!
//! 5. **OCR vocab projection IBP**: Linear(HIDDEN_DIM, VOCAB_SIZE) for OCR
//!    vocabulary logits.
//!
//! 6. **CTC blank probability IBP**: Softmax output, verify blank token
//!    probability bounded in [0, 1].
//!
//! 7. **CTC softmax output IBP**: Full CTC head: Linear -> Softmax, output
//!    character probabilities in [0, 1].
//!
//! 8. **OCR pipeline IBP**: Patch embed -> encoder -> CTC head -> softmax.
//!    End-to-end bounds from image pixels to character probabilities.
//!
//! 9. **RMSNorm IBP**: RMSNorm at 1536 dims (2B-scale normalization).
//!
//! 10. **SwiGLU FFN CROWN**: SwiGLU FFN with CROWN bounds through the
//!     gate_proj -> SiLU -> mul(up_proj) -> down_proj path.
//!
//! 11. **Two-layer encoder IBP**: 2 chained encoder layers verifying bounds
//!     propagation through repeated attention + FFN blocks.
//!
//! 12. **Line detection sigmoid IBP**: Sigmoid output for line detection
//!     bounding box confidence, bounded in [0, 1].
//!
//! 13. **Three-layer encoder IBP**: 3 chained encoder layers for deeper
//!     composition, testing bound widening across multiple attention + FFN
//!     blocks.
//!
//! 14. **Three-layer encoder CROWN**: Same 3-layer encoder with CROWN
//!     linearization for tighter bounds through deep stacks.
//!
//! 15. **Deep CTC head IBP**: Vocab projection -> softmax with multi-step
//!     verification asserting [0, 1] probability bounds.
//!
//! 16. **Full encoder -> CTC pipeline IBP**: Patch embed -> 2 encoder layers
//!     -> CTC projection -> softmax. End-to-end from image to character probs.
//!
//! 17. **Attention + RMSNorm + SwiGLU fusion composition IBP**: Fused
//!     sub-block: RMSNorm -> Attention -> RMSNorm -> SwiGLU without residual.
//!
//! 18. **Line detection branch IBP**: Encoder features -> linear projection
//!     -> sigmoid confidence map. Multi-layer detection head.
//!
//! 19. **Multi-head attention at 2B-scale dims CROWN**: 12-head attention
//!     with CROWN bounds at 1536-scale dimensions (scaled down to 48).
//!
//! 20. **RMSNorm -> SwiGLU -> RMSNorm sandwich IBP**: Pre-norm FFN sandwich
//!     testing normalization stability through SwiGLU gating.
//!
//! 21. **Patch embedding + positional encoding IBP**: Patch embed followed
//!     by additive positional encoding via learned bias.
//!
//! 22. **End-to-end: patch -> encoder stack -> CTC -> character probabilities
//!     CROWN**: Full pipeline with CROWN for tighter end-to-end bounds.
//!
//! 23. **Two-layer encoder CROWN**: 2 chained encoder layers with CROWN
//!     linearization through stacked RMSNorm layers.
//!
//! 24. **Two-layer encoder -> CTC head IBP**: 2 encoder layers -> Linear ->
//!     Softmax. Tests deep composition ending in probability bounds.
//!
//! 25-34: Multi-head CTC, 4/8-layer encoder stacks, residual accumulation,
//!     RMSNorm+SwiGLU CROWN, cross-attention, embedding->encoder->CTC,
//!     batched encoder, padding invariance.
//!
//! 35. **Deep 8-layer encoder CROWN**: CROWN linearization at 2B-representative
//!     depth through 8 stacked RMSNorm + attention + SwiGLU blocks.
//!
//! 36. **Large 24-head attention at 2048 dims (scaled down) IBP**: 24-head
//!     attention at wider hidden dimension (72 dims).
//!
//! 37. **SwiGLU FFN at 5632 intermediate dims (scaled down) IBP**: SwiGLU
//!     gating at 128-dim intermediate (representative of 5632 in production).
//!
//! 38. **Residual accumulation through 8+ layers IBP**: Pure 8-deep residual
//!     chain isolating residual contribution to bound widening.
//!
//! 39. **RMSNorm stability at large dimensions IBP**: RMSNorm at 72 dims
//!     testing stability at wider dimension than default.
//!
//! 40. **RMSNorm stability at large dimensions CROWN**: CROWN through
//!     RMSNorm at 72 dims for tighter bounds via linearization.
//!
//! 41. **Full CTC pipeline IBP**: Encoder -> linear -> softmax -> log_softmax
//!     for CTC decoding confidence (log-probabilities).
//!
//! 42. **Blank token probability bounds IBP**: CTC pipeline with ReLU
//!     intermediate, verifying blank token (index 0) bounded per timestep.
//!
//! 43. **Multi-character decoding across 65536 vocab (scaled down) IBP**:
//!     CTC head with 512-vocab softmax for large-vocab bounds.
//!
//! 44. **CTC prefix beam search monotonicity IBP**: Softmax per-element
//!     bounds verification for sum-to-one consistency.
//!
//! 45. **Patch embed -> 8-layer encoder -> CTC end-to-end IBP**: Deepest
//!     full pipeline from image pixels to character probabilities.
//!
//! 46. **Line detection + recognition composition IBP**: Two-branch encoder
//!     with detection (sigmoid) and recognition (softmax) heads.
//!
//! 47. **Multi-page document bounds IBP**: Encoder over concatenated patch
//!     sequences from 3 pages (12 patches total).
//!
//! 48. **Resolution scaling: patch embed at larger resolution IBP**: Patch
//!     embedding at 42x42 (9 patches) vs default 28x28 (4 patches).
//!
//! 49. **Resolution scaling: large resolution -> encoder -> CTC IBP**: Full
//!     pipeline at 42x42 resolution through encoder and CTC head.
//!
//! 50. **Full OCR end-to-end CROWN**: CROWN through patch -> 8-layer
//!     encoder -> CTC softmax for tighter end-to-end bounds.
//!
//! 51. **Four-layer encoder CROWN**: CROWN at medium depth (4 layers),
//!     bridging 2-layer (test 23) and 8-layer (test 35) CROWN tests.
//!
//! 52. **Eight-layer encoder -> CTC pipeline IBP**: Deepest encoder-CTC
//!     composition with 8 encoder layers + CTC softmax.
//!
//! Architecture references:
//! - FireRed-OCR: Qwen3-VL-2B variant for document OCR with CTC decoding
//! - Qwen2-VL / Qwen3-VL (Alibaba): Vision-language model with patch embedding,
//!   RMSNorm, SwiGLU, and multi-head attention
//! - RMSNorm (Zhang & Sennrich, 2019): replaces LayerNorm in Qwen
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN
//! - CTC (Graves et al. 2006): Connectionist Temporal Classification
//!
//! Dimensions (small for fast verification, structurally representative):
//! - IMG_SIZE=28, PATCH_SIZE=14, HIDDEN_DIM=48, FFN_DIM=96, SEQ_LEN=4,
//!   NUM_HEADS=12, HEAD_DIM=4, VOCAB_SIZE=256, LINE_DET_CH=32
//!
//! Part of #3906: NY compose tests for FireRed-OCR subgraphs.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// of Qwen3-VL-2B architecture used in FireRed-OCR
// ---------------------------------------------------------------------------

/// Image height and width (square image).
const IMG_SIZE: usize = 28;
/// Patch size (P). IMG_SIZE must be divisible by PATCH_SIZE.
const PATCH_SIZE: usize = 14;
/// Number of patches per spatial dimension.
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 2
/// Total number of patches.
const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE; // 4
/// Input channels (RGB).
const IN_CHANNELS: usize = 3;
/// Hidden dimension (scaled down from 1536 for test tractability).
const HIDDEN_DIM: usize = 48;
/// FFN intermediate dimension (SwiGLU gate and up projections).
const FFN_DIM: usize = 96;
/// Sequence length for encoder/decoder sub-block tests.
const SEQ_LEN: usize = 4;
/// Number of attention heads (12-head for 2B scale, scaled down).
const NUM_HEADS: usize = 12;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 4
/// OCR vocabulary size (characters + blank token for CTC).
const VOCAB_SIZE: usize = 256;
/// Line detection head channel count.
const LINE_DET_CH: usize = 32;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// 1. Patch embedding IBP: Conv2d(3, D, 14, stride=14)
// ===========================================================================

/// Build a FireRed-OCR patch embedding kernel.
///
/// Conv2d(3, HIDDEN_DIM, PATCH_SIZE, stride=PATCH_SIZE) maps image patches
/// to the hidden dimension. Reshape and transpose produce a sequence of
/// patch embeddings.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels in [0, 1]).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
fn build_firered_patch_embedding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_patch_embedding");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let weight = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let bias = b.add_input("patch_bias", &[HIDDEN_DIM]);

    // Conv2d: [3, 28, 28] -> [D, 2, 2]
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
        .expect("valid FireRed-OCR patch embedding kernel")
}

/// Image-domain input bounds: pixels in [0, 1].
fn image_bounds_01(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Bindings for patch embedding.
fn firered_patch_embedding_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // image [3, 28, 28]
        TensorParamBinding::ConstantTensor(w),    // patch_weight [D, 3, P, P]
        TensorParamBinding::ConstantTensor(bias), // patch_bias [D]
    ]
}

/// IBP bounds propagate through FireRed-OCR patch embedding.
#[test]
fn test_firered_patch_embedding_ibp() {
    let def = build_firered_patch_embedding_kernel();
    let bindings = firered_patch_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR patch embedding");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "output shape should be [NUM_PATCHES={NUM_PATCHES}, HIDDEN_DIM={HIDDEN_DIM}], got {:?}",
        lo.shape()
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR patch embedding IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// 2. Small attention IBP: 12-head attention at 1536 dims (scaled down)
// ===========================================================================

/// Build a 12-head self-attention kernel for FireRed-OCR.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, patch features).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_firered_small_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_small_attention");

    let input = b.add_input("patch_features", &[SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let shape = [SEQ_LEN, HIDDEN_DIM];

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // Residual connection
    let result = b.add_binary_add(input, attn_out, &shape);

    b.build(result)
        .expect("valid FireRed-OCR small attention kernel")
}

/// Bindings for small attention.
fn firered_small_attention_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,              // patch_features
        TensorParamBinding::ConstantTensor(q_w),   // q_proj_weight
        TensorParamBinding::ConstantTensor(k_w),   // k_proj_weight
        TensorParamBinding::ConstantTensor(v_w),   // v_proj_weight
        TensorParamBinding::ConstantTensor(out_w), // out_proj_weight
    ]
}

/// IBP bounds propagate through 12-head self-attention.
#[test]
fn test_firered_small_attention_ibp() {
    let def = build_firered_small_attention_kernel();
    let bindings = firered_small_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR small attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "small attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR small attention IBP (patches [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Residual connection preserves bounded output
    assert!(
        lo_min > -100.0,
        "small attention lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 3. Encoder layer IBP: RMSNorm -> Attention -> residual -> SwiGLU FFN
// ===========================================================================

/// Build a FireRed-OCR encoder layer.
///
/// RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_firered_encoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_encoder_layer");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Pre-attention RMSNorm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // Self-attention
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // Residual after attention
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

    // Residual after FFN
    let out = b.add_binary_add(residual1, ffn_out, &shape);

    b.build(out)
        .expect("valid FireRed-OCR encoder layer kernel")
}

/// Bindings for encoder layer.
fn firered_encoder_layer_bindings() -> Vec<TensorParamBinding> {
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

/// IBP bounds propagate through FireRed-OCR encoder layer.
#[test]
fn test_firered_encoder_layer_ibp() {
    let def = build_firered_encoder_layer_kernel();
    let bindings = firered_encoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR encoder layer");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "encoder layer output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR encoder layer IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 4. Encoder layer CROWN: Same with CROWN linearization
// ===========================================================================

/// CROWN bounds propagate through FireRed-OCR encoder layer.
///
/// RMSNorm requires CROWN linearization via IbpValidated mode. SwiGLU
/// multiplicative gating uses McCormick envelopes.
#[test]
fn test_firered_encoder_layer_crown() {
    let def = build_firered_encoder_layer_kernel();
    let bindings = firered_encoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR encoder layer CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record encoder layer.
#[test]
fn test_firered_encoder_layer_verify_and_record() {
    let def = build_firered_encoder_layer_kernel();
    let bindings = firered_encoder_layer_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "firered_ocr_encoder_layer");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 5. OCR vocab projection IBP: Linear(HIDDEN_DIM, VOCAB_SIZE)
// ===========================================================================

/// Build an OCR vocabulary projection kernel.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (logits per timestep).
fn build_firered_ocr_vocab_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_vocab_projection");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let weight = b.add_input("vocab_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let bias = b.add_input("vocab_bias", &[VOCAB_SIZE]);

    let out = b.add_linear(input, weight, Some(bias), &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid FireRed-OCR vocab projection kernel")
}

/// Bindings for OCR vocab projection.
fn firered_ocr_vocab_projection_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // encoder_output
        TensorParamBinding::ConstantTensor(w),    // vocab_weight
        TensorParamBinding::ConstantTensor(bias), // vocab_bias
    ]
}

/// IBP bounds propagate through OCR vocabulary projection.
///
/// Pure linear layer: output bounds scale with weight * input range.
/// With 0.02 weights, [-2, 2] input, D=48: max output ~= 0.02 * 48 * 2 = 1.92.
#[test]
fn test_firered_ocr_vocab_projection_ibp() {
    let def = build_firered_ocr_vocab_projection_kernel();
    let bindings = firered_ocr_vocab_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR vocab projection");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "vocab projection output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR vocab projection IBP (encoder [-2,2]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Linear with D=48, weight=0.02, input in [-2, 2]:
    // max output = sum(|w_i| * 2.0) = 48 * 0.02 * 2 = 1.92
    assert!(
        hi_max < 10.0,
        "vocab projection upper should be < 10 with small weights, got {hi_max}"
    );
}

// ===========================================================================
// 6. CTC blank probability IBP: Softmax output, blank token in [0, 1]
// ===========================================================================

/// Build a CTC blank probability kernel.
///
/// Softmax over vocabulary dimension. The blank token (index 0 by CTC
/// convention) probability is bounded in [0, 1].
///
/// Input: `[SEQ_LEN, VOCAB_SIZE]` (Variable, logits).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probabilities in [0, 1]).
fn build_firered_ctc_blank_probability_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_ctc_blank_probability");

    let input = b.add_input("logits", &[SEQ_LEN, VOCAB_SIZE]);

    let out = b.add_softmax(input, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid FireRed-OCR CTC blank probability kernel")
}

/// Bindings for CTC blank probability (pure softmax on variable input).
fn firered_ctc_blank_probability_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable] // logits
}

/// IBP bounds through CTC softmax: all outputs must be in [0, 1].
#[test]
fn test_firered_ctc_blank_probability_ibp() {
    let def = build_firered_ctc_blank_probability_kernel();
    let bindings = firered_ctc_blank_probability_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, VOCAB_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR CTC blank probability");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "CTC blank probability output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR CTC blank probability IBP (logits [-2,2]): bounds=[{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "softmax output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "softmax output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 7. CTC softmax output IBP: Linear -> Softmax
// ===========================================================================

/// Build a full CTC head: Linear projection + Softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder output).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities in [0, 1]).
fn build_firered_ctc_softmax_output_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_ctc_softmax_output");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let weight = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    // Linear projection to logits
    let logits = b.add_linear(input, weight, Some(bias), &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax over vocabulary dimension
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid FireRed-OCR CTC softmax output kernel")
}

/// Bindings for CTC softmax output.
fn firered_ctc_softmax_output_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // encoder_output
        TensorParamBinding::ConstantTensor(w),    // ctc_weight
        TensorParamBinding::ConstantTensor(bias), // ctc_bias
    ]
}

/// IBP bounds through full CTC head: output must be in [0, 1].
#[test]
fn test_firered_ctc_softmax_output_ibp() {
    let def = build_firered_ctc_softmax_output_kernel();
    let bindings = firered_ctc_softmax_output_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR CTC softmax output");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "CTC softmax output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR CTC softmax output IBP (encoder [-2,2]): bounds=[{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "softmax output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "softmax output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 8. OCR pipeline IBP: patch embed -> encoder -> CTC head -> softmax
// ===========================================================================

/// Build the end-to-end FireRed-OCR pipeline.
///
/// Patch embedding -> encoder layer (RMSNorm + Attention + SwiGLU) ->
/// CTC linear head -> softmax.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[NUM_PATCHES, VOCAB_SIZE]` (character probabilities).
fn build_firered_ocr_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_pipeline");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];
    let ffn_shape = [NUM_PATCHES, FFN_DIM];

    // --- Patch embedding ---
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_bias = b.add_input("patch_bias", &[HIDDEN_DIM]);

    let conv_out = b.add_conv2d(
        input,
        patch_w,
        Some(patch_bias),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &patch_shape);

    // --- Encoder layer ---
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(patches, norm1_eps, 1, norm1_w, &patch_shape);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &patch_shape);
    let k = b.add_linear(normed1, k_w, None, &patch_shape);
    let v = b.add_linear(normed1, v_w, None, &patch_shape);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &patch_shape);
    let attn_out = b.add_linear(attn, out_w, None, &patch_shape);
    let residual1 = b.add_binary_add(patches, attn_out, &patch_shape);

    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(residual1, norm2_eps, 1, norm2_w, &patch_shape);

    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &patch_shape);
    let enc_out = b.add_binary_add(residual1, ffn_out, &patch_shape);

    // --- CTC head + softmax ---
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(enc_out, ctc_w, Some(ctc_bias), &[NUM_PATCHES, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[NUM_PATCHES, VOCAB_SIZE]);

    b.build(out).expect("valid FireRed-OCR pipeline kernel")
}

/// Bindings for full OCR pipeline.
fn firered_ocr_pipeline_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                       // image
        TensorParamBinding::ConstantTensor(patch_w),        // patch_weight
        TensorParamBinding::ConstantTensor(patch_bias),     // patch_bias
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(qkvo_w),         // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
        TensorParamBinding::ConstantTensor(ctc_w),          // ctc_weight
        TensorParamBinding::ConstantTensor(ctc_bias),       // ctc_bias
    ]
}

/// IBP through full OCR pipeline: image [0,1] -> character probabilities.
#[test]
fn test_firered_ocr_pipeline_ibp() {
    let def = build_firered_ocr_pipeline_kernel();
    let bindings = firered_ocr_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR pipeline");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, VOCAB_SIZE],
        "OCR pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR pipeline IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    // End-to-end: softmax clamps to [0, 1]
    assert!(
        lo_min >= -1e-4,
        "OCR pipeline output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "OCR pipeline output upper should be <= 1, got {hi_max}"
    );
}

/// Verify and record full OCR pipeline.
#[test]
fn test_firered_ocr_pipeline_verify_and_record() {
    let def = build_firered_ocr_pipeline_kernel();
    let bindings = firered_ocr_pipeline_bindings();
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let result = verify_and_assert(&def, &bindings, &input, "firered_ocr_pipeline");
    assert_eq!(result.num_variables, 1, "single Variable input (image)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, VOCAB_SIZE]);
}

// ===========================================================================
// 9. RMSNorm IBP at 2B-scale dimensions
// ===========================================================================

/// Build an RMSNorm kernel for FireRed-OCR (2B scale).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_firered_rmsnorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_rmsnorm");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[HIDDEN_DIM]);

    let out = b.add_rms_norm(input, eps, 1, weight, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid FireRed-OCR RMSNorm kernel")
}

/// Bindings for RMSNorm.
fn firered_rmsnorm_bindings() -> Vec<TensorParamBinding> {
    let weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(weight),
    ]
}

/// IBP bounds propagate through FireRed-OCR RMSNorm.
#[test]
fn test_firered_rmsnorm_ibp() {
    let def = build_firered_rmsnorm_kernel();
    let bindings = firered_rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR RMSNorm");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "RMSNorm output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR RMSNorm IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record FireRed-OCR RMSNorm.
#[test]
fn test_firered_rmsnorm_verify_and_record() {
    let def = build_firered_rmsnorm_kernel();
    let bindings = firered_rmsnorm_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "firered_ocr_rmsnorm");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 10. SwiGLU FFN CROWN
// ===========================================================================

/// Build a SwiGLU FFN kernel for FireRed-OCR.
///
/// gate_proj -> SiLU -> mul(up_proj) -> down_proj.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_firered_swiglu_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_swiglu_ffn");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let gate_w = b.add_input("gate_proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_proj_weight", &[HIDDEN_DIM, FFN_DIM]);

    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // Gate branch: gate_proj -> SiLU (x * sigmoid(x))
    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);

    // Up branch: up_proj
    let up = b.add_linear(input, up_w, None, &ffn_shape);

    // Multiplicative gating
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);

    // Down projection
    let out = b.add_linear(hidden, down_w, None, &out_shape);

    b.build(out).expect("valid FireRed-OCR SwiGLU FFN kernel")
}

/// Bindings for SwiGLU FFN.
fn firered_swiglu_ffn_bindings() -> Vec<TensorParamBinding> {
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(gate_w),
        TensorParamBinding::ConstantTensor(up_w),
        TensorParamBinding::ConstantTensor(down_w),
    ]
}

/// CROWN bounds propagate through SwiGLU FFN.
///
/// SiLU gating (x * sigmoid(x)) requires McCormick envelopes for the
/// bilinear term. CROWN linearization produces tighter bounds than IBP
/// through the multiplicative interactions.
#[test]
fn test_firered_swiglu_ffn_crown() {
    let def = build_firered_swiglu_ffn_kernel();
    let bindings = firered_swiglu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR SwiGLU FFN CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record SwiGLU FFN.
#[test]
fn test_firered_swiglu_ffn_verify_and_record() {
    let def = build_firered_swiglu_ffn_kernel();
    let bindings = firered_swiglu_ffn_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "firered_ocr_swiglu_ffn");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 11. Two-layer encoder IBP: 2 chained encoder layers
// ===========================================================================

/// Build two stacked FireRed-OCR encoder layers.
///
/// Verifies bounds propagation through repeated attention + FFN layers.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_firered_two_layer_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_two_layer_encoder");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // --- Block 1 ---
    let b1_norm1_eps = b.add_input("b1_norm1_eps", &[1]);
    let b1_norm1_w = b.add_input("b1_norm1_weight", &[HIDDEN_DIM]);
    let b1_normed1 = b.add_rms_norm(input, b1_norm1_eps, 1, b1_norm1_w, &shape);

    let b1_q_w = b.add_input("b1_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_k_w = b.add_input("b1_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_v_w = b.add_input("b1_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_out_w = b.add_input("b1_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let b1_q = b.add_linear(b1_normed1, b1_q_w, None, &shape);
    let b1_k = b.add_linear(b1_normed1, b1_k_w, None, &shape);
    let b1_v = b.add_linear(b1_normed1, b1_v_w, None, &shape);
    let b1_attn = b.add_attention(
        b1_q,
        b1_k,
        b1_v,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );
    let b1_attn_out = b.add_linear(b1_attn, b1_out_w, None, &shape);
    let b1_res1 = b.add_binary_add(input, b1_attn_out, &shape);

    let b1_norm2_eps = b.add_input("b1_norm2_eps", &[1]);
    let b1_norm2_w = b.add_input("b1_norm2_weight", &[HIDDEN_DIM]);
    let b1_normed2 = b.add_rms_norm(b1_res1, b1_norm2_eps, 1, b1_norm2_w, &shape);

    let b1_gate_w = b.add_input("b1_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b1_up_w = b.add_input("b1_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b1_down_w = b.add_input("b1_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let b1_gate = b.add_linear(b1_normed2, b1_gate_w, None, &ffn_shape);
    let b1_gate_sig = b.add_sigmoid(b1_gate, &ffn_shape);
    let b1_gate_act = b.add_binary_mul(b1_gate, b1_gate_sig, &ffn_shape);
    let b1_up = b.add_linear(b1_normed2, b1_up_w, None, &ffn_shape);
    let b1_hidden = b.add_binary_mul(b1_gate_act, b1_up, &ffn_shape);
    let b1_ffn_out = b.add_linear(b1_hidden, b1_down_w, None, &shape);
    let b1_res2 = b.add_binary_add(b1_res1, b1_ffn_out, &shape);

    // --- Block 2 ---
    let b2_norm1_eps = b.add_input("b2_norm1_eps", &[1]);
    let b2_norm1_w = b.add_input("b2_norm1_weight", &[HIDDEN_DIM]);
    let b2_normed1 = b.add_rms_norm(b1_res2, b2_norm1_eps, 1, b2_norm1_w, &shape);

    let b2_q_w = b.add_input("b2_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_k_w = b.add_input("b2_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_v_w = b.add_input("b2_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_out_w = b.add_input("b2_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let b2_q = b.add_linear(b2_normed1, b2_q_w, None, &shape);
    let b2_k = b.add_linear(b2_normed1, b2_k_w, None, &shape);
    let b2_v = b.add_linear(b2_normed1, b2_v_w, None, &shape);
    let b2_attn = b.add_attention(
        b2_q,
        b2_k,
        b2_v,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );
    let b2_attn_out = b.add_linear(b2_attn, b2_out_w, None, &shape);
    let b2_res1 = b.add_binary_add(b1_res2, b2_attn_out, &shape);

    let b2_norm2_eps = b.add_input("b2_norm2_eps", &[1]);
    let b2_norm2_w = b.add_input("b2_norm2_weight", &[HIDDEN_DIM]);
    let b2_normed2 = b.add_rms_norm(b2_res1, b2_norm2_eps, 1, b2_norm2_w, &shape);

    let b2_gate_w = b.add_input("b2_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b2_up_w = b.add_input("b2_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b2_down_w = b.add_input("b2_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let b2_gate = b.add_linear(b2_normed2, b2_gate_w, None, &ffn_shape);
    let b2_gate_sig = b.add_sigmoid(b2_gate, &ffn_shape);
    let b2_gate_act = b.add_binary_mul(b2_gate, b2_gate_sig, &ffn_shape);
    let b2_up = b.add_linear(b2_normed2, b2_up_w, None, &ffn_shape);
    let b2_hidden = b.add_binary_mul(b2_gate_act, b2_up, &ffn_shape);
    let b2_ffn_out = b.add_linear(b2_hidden, b2_down_w, None, &shape);
    let out = b.add_binary_add(b2_res1, b2_ffn_out, &shape);

    b.build(out)
        .expect("valid FireRed-OCR two-layer encoder kernel")
}

/// Bindings for two-layer encoder (2 blocks).
fn firered_two_layer_encoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    // Build bindings for 2 identical blocks
    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _block in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gate_weight
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // up_weight
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // down_weight
    }

    bindings
}

/// IBP bounds propagate through 2-block encoder stack.
#[test]
fn test_firered_two_layer_encoder_ibp() {
    let def = build_firered_two_layer_encoder_kernel();
    let bindings = firered_two_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR two-layer encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "two-layer encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR two-layer encoder IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 12. Line detection sigmoid IBP: Sigmoid for line detection confidence
// ===========================================================================

/// Build a line detection sigmoid output kernel.
///
/// FireRed-OCR includes a line detection head that outputs bounding box
/// confidence scores via sigmoid activation. Similar to a DB-style
/// binarization head but for text line detection rather than word detection.
///
/// Input: `[LINE_DET_CH, IMG_SIZE, IMG_SIZE]` (Variable, feature map).
/// Output: `[1, IMG_SIZE, IMG_SIZE]` (confidence map in [0, 1]).
fn build_firered_line_detection_sigmoid_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_line_detection_sigmoid");

    let input = b.add_input("features", &[LINE_DET_CH, IMG_SIZE, IMG_SIZE]);
    let conv_w = b.add_input("det_weight", &[1, LINE_DET_CH, 1, 1]);
    let conv_bias = b.add_input("det_bias", &[1]);

    let proj_shape = [1, IMG_SIZE, IMG_SIZE];

    // 1x1 conv projection: [LINE_DET_CH, 28, 28] -> [1, 28, 28]
    let proj = b.add_conv2d(input, conv_w, Some(conv_bias), 1, 1, 0, 0, &proj_shape);

    // Sigmoid: output confidence in [0, 1]
    let out = b.add_sigmoid(proj, &proj_shape);

    b.build(out)
        .expect("valid FireRed-OCR line detection sigmoid kernel")
}

/// Bindings for line detection sigmoid.
fn firered_line_detection_sigmoid_bindings() -> Vec<TensorParamBinding> {
    let conv_w = ArrayD::from_elem(IxDyn(&[1, LINE_DET_CH, 1, 1]), WEIGHT_MAG);
    let conv_bias = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                  // features
        TensorParamBinding::ConstantTensor(conv_w),    // det_weight
        TensorParamBinding::ConstantTensor(conv_bias), // det_bias
    ]
}

/// IBP bounds through line detection sigmoid: result must be in [0, 1].
#[test]
fn test_firered_line_detection_sigmoid_ibp() {
    let def = build_firered_line_detection_sigmoid_kernel();
    let bindings = firered_line_detection_sigmoid_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[LINE_DET_CH, IMG_SIZE, IMG_SIZE], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR line detection sigmoid");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[1, IMG_SIZE, IMG_SIZE],
        "line detection sigmoid output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR line detection sigmoid IBP (features [-2,2]): bounds=[{lo_min}, {hi_max}]"
    );

    // Sigmoid output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "sigmoid output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "sigmoid output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 13. Three-layer encoder IBP: 3 chained encoder layers
// ===========================================================================

/// Build three stacked FireRed-OCR encoder layers.
///
/// Tests bounds propagation through a deeper encoder stack (3 blocks).
/// Each block: RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_firered_three_layer_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_three_layer_encoder");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // --- Block 1 ---
    let b1_norm1_eps = b.add_input("b1_norm1_eps", &[1]);
    let b1_norm1_w = b.add_input("b1_norm1_weight", &[HIDDEN_DIM]);
    let b1_normed1 = b.add_rms_norm(input, b1_norm1_eps, 1, b1_norm1_w, &shape);

    let b1_q_w = b.add_input("b1_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_k_w = b.add_input("b1_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_v_w = b.add_input("b1_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_out_w = b.add_input("b1_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let b1_q = b.add_linear(b1_normed1, b1_q_w, None, &shape);
    let b1_k = b.add_linear(b1_normed1, b1_k_w, None, &shape);
    let b1_v = b.add_linear(b1_normed1, b1_v_w, None, &shape);
    let b1_attn = b.add_attention(
        b1_q,
        b1_k,
        b1_v,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );
    let b1_attn_out = b.add_linear(b1_attn, b1_out_w, None, &shape);
    let b1_res1 = b.add_binary_add(input, b1_attn_out, &shape);

    let b1_norm2_eps = b.add_input("b1_norm2_eps", &[1]);
    let b1_norm2_w = b.add_input("b1_norm2_weight", &[HIDDEN_DIM]);
    let b1_normed2 = b.add_rms_norm(b1_res1, b1_norm2_eps, 1, b1_norm2_w, &shape);

    let b1_gate_w = b.add_input("b1_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b1_up_w = b.add_input("b1_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b1_down_w = b.add_input("b1_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let b1_gate = b.add_linear(b1_normed2, b1_gate_w, None, &ffn_shape);
    let b1_gate_sig = b.add_sigmoid(b1_gate, &ffn_shape);
    let b1_gate_act = b.add_binary_mul(b1_gate, b1_gate_sig, &ffn_shape);
    let b1_up = b.add_linear(b1_normed2, b1_up_w, None, &ffn_shape);
    let b1_hidden = b.add_binary_mul(b1_gate_act, b1_up, &ffn_shape);
    let b1_ffn_out = b.add_linear(b1_hidden, b1_down_w, None, &shape);
    let b1_res2 = b.add_binary_add(b1_res1, b1_ffn_out, &shape);

    // --- Block 2 ---
    let b2_norm1_eps = b.add_input("b2_norm1_eps", &[1]);
    let b2_norm1_w = b.add_input("b2_norm1_weight", &[HIDDEN_DIM]);
    let b2_normed1 = b.add_rms_norm(b1_res2, b2_norm1_eps, 1, b2_norm1_w, &shape);

    let b2_q_w = b.add_input("b2_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_k_w = b.add_input("b2_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_v_w = b.add_input("b2_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_out_w = b.add_input("b2_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let b2_q = b.add_linear(b2_normed1, b2_q_w, None, &shape);
    let b2_k = b.add_linear(b2_normed1, b2_k_w, None, &shape);
    let b2_v = b.add_linear(b2_normed1, b2_v_w, None, &shape);
    let b2_attn = b.add_attention(
        b2_q,
        b2_k,
        b2_v,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );
    let b2_attn_out = b.add_linear(b2_attn, b2_out_w, None, &shape);
    let b2_res1 = b.add_binary_add(b1_res2, b2_attn_out, &shape);

    let b2_norm2_eps = b.add_input("b2_norm2_eps", &[1]);
    let b2_norm2_w = b.add_input("b2_norm2_weight", &[HIDDEN_DIM]);
    let b2_normed2 = b.add_rms_norm(b2_res1, b2_norm2_eps, 1, b2_norm2_w, &shape);

    let b2_gate_w = b.add_input("b2_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b2_up_w = b.add_input("b2_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b2_down_w = b.add_input("b2_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let b2_gate = b.add_linear(b2_normed2, b2_gate_w, None, &ffn_shape);
    let b2_gate_sig = b.add_sigmoid(b2_gate, &ffn_shape);
    let b2_gate_act = b.add_binary_mul(b2_gate, b2_gate_sig, &ffn_shape);
    let b2_up = b.add_linear(b2_normed2, b2_up_w, None, &ffn_shape);
    let b2_hidden = b.add_binary_mul(b2_gate_act, b2_up, &ffn_shape);
    let b2_ffn_out = b.add_linear(b2_hidden, b2_down_w, None, &shape);
    let b2_res2 = b.add_binary_add(b2_res1, b2_ffn_out, &shape);

    // --- Block 3 ---
    let b3_norm1_eps = b.add_input("b3_norm1_eps", &[1]);
    let b3_norm1_w = b.add_input("b3_norm1_weight", &[HIDDEN_DIM]);
    let b3_normed1 = b.add_rms_norm(b2_res2, b3_norm1_eps, 1, b3_norm1_w, &shape);

    let b3_q_w = b.add_input("b3_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b3_k_w = b.add_input("b3_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b3_v_w = b.add_input("b3_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b3_out_w = b.add_input("b3_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let b3_q = b.add_linear(b3_normed1, b3_q_w, None, &shape);
    let b3_k = b.add_linear(b3_normed1, b3_k_w, None, &shape);
    let b3_v = b.add_linear(b3_normed1, b3_v_w, None, &shape);
    let b3_attn = b.add_attention(
        b3_q,
        b3_k,
        b3_v,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );
    let b3_attn_out = b.add_linear(b3_attn, b3_out_w, None, &shape);
    let b3_res1 = b.add_binary_add(b2_res2, b3_attn_out, &shape);

    let b3_norm2_eps = b.add_input("b3_norm2_eps", &[1]);
    let b3_norm2_w = b.add_input("b3_norm2_weight", &[HIDDEN_DIM]);
    let b3_normed2 = b.add_rms_norm(b3_res1, b3_norm2_eps, 1, b3_norm2_w, &shape);

    let b3_gate_w = b.add_input("b3_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b3_up_w = b.add_input("b3_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b3_down_w = b.add_input("b3_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let b3_gate = b.add_linear(b3_normed2, b3_gate_w, None, &ffn_shape);
    let b3_gate_sig = b.add_sigmoid(b3_gate, &ffn_shape);
    let b3_gate_act = b.add_binary_mul(b3_gate, b3_gate_sig, &ffn_shape);
    let b3_up = b.add_linear(b3_normed2, b3_up_w, None, &ffn_shape);
    let b3_hidden = b.add_binary_mul(b3_gate_act, b3_up, &ffn_shape);
    let b3_ffn_out = b.add_linear(b3_hidden, b3_down_w, None, &shape);
    let out = b.add_binary_add(b3_res1, b3_ffn_out, &shape);

    b.build(out)
        .expect("valid FireRed-OCR three-layer encoder kernel")
}

/// Bindings for three-layer encoder (3 blocks).
fn firered_three_layer_encoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _block in 0..3 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gate_weight
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // up_weight
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // down_weight
    }

    bindings
}

/// IBP bounds propagate through 3-block encoder stack.
///
/// Verifies that bounds remain finite and non-degenerate even after 3
/// repeated attention + SwiGLU FFN layers. Tests bound widening behavior
/// across deeper compositions.
#[test]
fn test_firered_three_layer_encoder_ibp() {
    let def = build_firered_three_layer_encoder_kernel();
    let bindings = firered_three_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR three-layer encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "three-layer encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR three-layer encoder IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 14. Three-layer encoder CROWN: CROWN through 3-layer encoder stack
// ===========================================================================

/// CROWN bounds propagate through 3-layer FireRed-OCR encoder.
///
/// CROWN linearization produces tighter bounds than IBP through the
/// deep encoder stack. RMSNorm layers require IbpValidated mode;
/// SwiGLU gating uses McCormick envelopes at each layer.
#[test]
fn test_firered_three_layer_encoder_crown() {
    let def = build_firered_three_layer_encoder_kernel();
    let bindings = firered_three_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR three-layer encoder CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record three-layer encoder.
#[test]
fn test_firered_three_layer_encoder_verify_and_record() {
    let def = build_firered_three_layer_encoder_kernel();
    let bindings = firered_three_layer_encoder_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "firered_ocr_three_layer_encoder");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 15. Deep CTC head IBP: vocab projection -> softmax with multi-step bounds
// ===========================================================================

/// Build a deep CTC head with an intermediate hidden layer.
///
/// Encoder output -> Linear(D, D) -> ReLU -> Linear(D, VOCAB) -> Softmax.
/// Tests multi-step bounds through activation + projection before softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities in [0, 1]).
fn build_firered_deep_ctc_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_deep_ctc_head");

    let input = b.add_input("encoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let hidden_w = b.add_input("ctc_hidden_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let hidden_bias = b.add_input("ctc_hidden_bias", &[HIDDEN_DIM]);
    let vocab_w = b.add_input("ctc_vocab_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let vocab_bias = b.add_input("ctc_vocab_bias", &[VOCAB_SIZE]);

    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Intermediate hidden layer with ReLU
    let hidden = b.add_linear(input, hidden_w, Some(hidden_bias), &shape);
    let activated = b.add_relu(hidden, &shape);

    // Vocab projection
    let logits = b.add_linear(activated, vocab_w, Some(vocab_bias), &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax over vocabulary
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid FireRed-OCR deep CTC head kernel")
}

/// Bindings for deep CTC head.
fn firered_deep_ctc_head_bindings() -> Vec<TensorParamBinding> {
    let hidden_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let hidden_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let vocab_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let vocab_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                    // encoder_output
        TensorParamBinding::ConstantTensor(hidden_w),    // ctc_hidden_weight
        TensorParamBinding::ConstantTensor(hidden_bias), // ctc_hidden_bias
        TensorParamBinding::ConstantTensor(vocab_w),     // ctc_vocab_weight
        TensorParamBinding::ConstantTensor(vocab_bias),  // ctc_vocab_bias
    ]
}

/// IBP bounds through deep CTC head: Linear -> ReLU -> Linear -> Softmax.
///
/// The intermediate ReLU clamps negatives, reducing bound widths before
/// the vocabulary projection. Output probabilities must be in [0, 1].
#[test]
fn test_firered_deep_ctc_head_ibp() {
    let def = build_firered_deep_ctc_head_kernel();
    let bindings = firered_deep_ctc_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR deep CTC head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "deep CTC head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR deep CTC head IBP (encoder [-2,2]): bounds=[{lo_min}, {hi_max}]");

    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "deep CTC head output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "deep CTC head output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 16. Full encoder -> CTC pipeline IBP: patch embed -> 2 encoder -> CTC
// ===========================================================================

/// Build the full encoder -> CTC pipeline with 2 encoder layers.
///
/// Patch embedding -> 2 encoder layers (RMSNorm + Attention + SwiGLU) ->
/// CTC Linear -> Softmax. Deeper than the single-layer pipeline in test 8.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image in [0, 1]).
/// Output: `[NUM_PATCHES, VOCAB_SIZE]` (character probabilities).
fn build_firered_full_encoder_ctc_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_full_encoder_ctc_pipeline");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];
    let ffn_shape = [NUM_PATCHES, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // --- Patch embedding ---
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_bias = b.add_input("patch_bias", &[HIDDEN_DIM]);

    let conv_out = b.add_conv2d(
        input,
        patch_w,
        Some(patch_bias),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let patches = b.add_transpose(reshaped, &[1, 0], &patch_shape);

    // --- Encoder layer 1 ---
    let e1_norm1_eps = b.add_input("e1_norm1_eps", &[1]);
    let e1_norm1_w = b.add_input("e1_norm1_weight", &[HIDDEN_DIM]);
    let e1_normed1 = b.add_rms_norm(patches, e1_norm1_eps, 1, e1_norm1_w, &patch_shape);

    let e1_q_w = b.add_input("e1_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let e1_k_w = b.add_input("e1_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let e1_v_w = b.add_input("e1_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let e1_out_w = b.add_input("e1_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let e1_q = b.add_linear(e1_normed1, e1_q_w, None, &patch_shape);
    let e1_k = b.add_linear(e1_normed1, e1_k_w, None, &patch_shape);
    let e1_v = b.add_linear(e1_normed1, e1_v_w, None, &patch_shape);
    let e1_attn = b.add_attention(
        e1_q,
        e1_k,
        e1_v,
        AttentionMask::Standard,
        Some(scale),
        &patch_shape,
    );
    let e1_attn_out = b.add_linear(e1_attn, e1_out_w, None, &patch_shape);
    let e1_res1 = b.add_binary_add(patches, e1_attn_out, &patch_shape);

    let e1_norm2_eps = b.add_input("e1_norm2_eps", &[1]);
    let e1_norm2_w = b.add_input("e1_norm2_weight", &[HIDDEN_DIM]);
    let e1_normed2 = b.add_rms_norm(e1_res1, e1_norm2_eps, 1, e1_norm2_w, &patch_shape);

    let e1_gate_w = b.add_input("e1_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let e1_up_w = b.add_input("e1_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let e1_down_w = b.add_input("e1_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let e1_gate = b.add_linear(e1_normed2, e1_gate_w, None, &ffn_shape);
    let e1_gate_sig = b.add_sigmoid(e1_gate, &ffn_shape);
    let e1_gate_act = b.add_binary_mul(e1_gate, e1_gate_sig, &ffn_shape);
    let e1_up = b.add_linear(e1_normed2, e1_up_w, None, &ffn_shape);
    let e1_hidden = b.add_binary_mul(e1_gate_act, e1_up, &ffn_shape);
    let e1_ffn_out = b.add_linear(e1_hidden, e1_down_w, None, &patch_shape);
    let e1_res2 = b.add_binary_add(e1_res1, e1_ffn_out, &patch_shape);

    // --- Encoder layer 2 ---
    let e2_norm1_eps = b.add_input("e2_norm1_eps", &[1]);
    let e2_norm1_w = b.add_input("e2_norm1_weight", &[HIDDEN_DIM]);
    let e2_normed1 = b.add_rms_norm(e1_res2, e2_norm1_eps, 1, e2_norm1_w, &patch_shape);

    let e2_q_w = b.add_input("e2_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let e2_k_w = b.add_input("e2_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let e2_v_w = b.add_input("e2_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let e2_out_w = b.add_input("e2_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let e2_q = b.add_linear(e2_normed1, e2_q_w, None, &patch_shape);
    let e2_k = b.add_linear(e2_normed1, e2_k_w, None, &patch_shape);
    let e2_v = b.add_linear(e2_normed1, e2_v_w, None, &patch_shape);
    let e2_attn = b.add_attention(
        e2_q,
        e2_k,
        e2_v,
        AttentionMask::Standard,
        Some(scale),
        &patch_shape,
    );
    let e2_attn_out = b.add_linear(e2_attn, e2_out_w, None, &patch_shape);
    let e2_res1 = b.add_binary_add(e1_res2, e2_attn_out, &patch_shape);

    let e2_norm2_eps = b.add_input("e2_norm2_eps", &[1]);
    let e2_norm2_w = b.add_input("e2_norm2_weight", &[HIDDEN_DIM]);
    let e2_normed2 = b.add_rms_norm(e2_res1, e2_norm2_eps, 1, e2_norm2_w, &patch_shape);

    let e2_gate_w = b.add_input("e2_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let e2_up_w = b.add_input("e2_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let e2_down_w = b.add_input("e2_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let e2_gate = b.add_linear(e2_normed2, e2_gate_w, None, &ffn_shape);
    let e2_gate_sig = b.add_sigmoid(e2_gate, &ffn_shape);
    let e2_gate_act = b.add_binary_mul(e2_gate, e2_gate_sig, &ffn_shape);
    let e2_up = b.add_linear(e2_normed2, e2_up_w, None, &ffn_shape);
    let e2_hidden = b.add_binary_mul(e2_gate_act, e2_up, &ffn_shape);
    let e2_ffn_out = b.add_linear(e2_hidden, e2_down_w, None, &patch_shape);
    let enc_out = b.add_binary_add(e2_res1, e2_ffn_out, &patch_shape);

    // --- CTC head + softmax ---
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(enc_out, ctc_w, Some(ctc_bias), &[NUM_PATCHES, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[NUM_PATCHES, VOCAB_SIZE]);

    b.build(out)
        .expect("valid FireRed-OCR full encoder CTC pipeline kernel")
}

/// Bindings for full encoder -> CTC pipeline.
fn firered_full_encoder_ctc_pipeline_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    let mut bindings = vec![
        TensorParamBinding::Variable,                   // image
        TensorParamBinding::ConstantTensor(patch_w),    // patch_weight
        TensorParamBinding::ConstantTensor(patch_bias), // patch_bias
    ];

    for _layer in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gate_weight
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // up_weight
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // down_weight
    }

    bindings.push(TensorParamBinding::ConstantTensor(ctc_w)); // ctc_weight
    bindings.push(TensorParamBinding::ConstantTensor(ctc_bias)); // ctc_bias

    bindings
}

/// IBP through full 2-layer encoder -> CTC pipeline: image -> char probs.
///
/// End-to-end from image pixels to character probabilities through 2 encoder
/// layers. Output must be in [0, 1] (softmax terminal).
#[test]
fn test_firered_full_encoder_ctc_pipeline_ibp() {
    let def = build_firered_full_encoder_ctc_pipeline_kernel();
    let bindings = firered_full_encoder_ctc_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR full encoder CTC pipeline");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, VOCAB_SIZE],
        "full encoder CTC pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR full encoder CTC pipeline IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]"
    );

    // Softmax terminal: output in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "full encoder CTC pipeline output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "full encoder CTC pipeline output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 17. Attention + RMSNorm + SwiGLU fusion composition IBP
// ===========================================================================

/// Build a fused attention + RMSNorm + SwiGLU sub-block (no residuals).
///
/// Tests direct composition without skip connections: RMSNorm -> Attention ->
/// RMSNorm -> SwiGLU FFN. Without residuals, bounds propagation is more
/// constrained by the normalization layers.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_firered_attention_norm_swiglu_fused_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_attn_norm_swiglu_fused");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // RMSNorm before attention
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // Attention
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // RMSNorm before FFN (no residual -- direct composition)
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(attn_out, norm2_eps, 1, norm2_w, &shape);

    // SwiGLU FFN
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &shape);

    b.build(out)
        .expect("valid FireRed-OCR attention+norm+SwiGLU fused kernel")
}

/// Bindings for fused attention + RMSNorm + SwiGLU sub-block.
fn firered_attention_norm_swiglu_fused_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(qkvo_w),         // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
    ]
}

/// IBP bounds through fused attention + RMSNorm + SwiGLU (no residuals).
#[test]
fn test_firered_attention_norm_swiglu_fused_ibp() {
    let def = build_firered_attention_norm_swiglu_fused_kernel();
    let bindings = firered_attention_norm_swiglu_fused_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR fused attn+norm+SwiGLU");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "fused attn+norm+SwiGLU output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR fused attn+norm+SwiGLU IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 18. Line detection branch IBP: encoder features -> linear -> sigmoid
// ===========================================================================

/// Build a multi-layer line detection head.
///
/// Encoder features -> Linear(D, LINE_DET_CH) -> ReLU -> Linear(LINE_DET_CH, 1)
/// -> Sigmoid. Tests deeper detection branch with activation.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, encoder features).
/// Output: `[SEQ_LEN, 1]` (line confidence scores in [0, 1]).
fn build_firered_line_detection_branch_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_line_detection_branch");

    let input = b.add_input("encoder_features", &[SEQ_LEN, HIDDEN_DIM]);
    let proj_w = b.add_input("det_proj_weight", &[LINE_DET_CH, HIDDEN_DIM]);
    let proj_bias = b.add_input("det_proj_bias", &[LINE_DET_CH]);
    let head_w = b.add_input("det_head_weight", &[1, LINE_DET_CH]);
    let head_bias = b.add_input("det_head_bias", &[1]);

    // Linear projection to detection channels
    let proj = b.add_linear(input, proj_w, Some(proj_bias), &[SEQ_LEN, LINE_DET_CH]);

    // ReLU activation
    let activated = b.add_relu(proj, &[SEQ_LEN, LINE_DET_CH]);

    // Final 1-channel head
    let logits = b.add_linear(activated, head_w, Some(head_bias), &[SEQ_LEN, 1]);

    // Sigmoid for confidence score
    let out = b.add_sigmoid(logits, &[SEQ_LEN, 1]);

    b.build(out)
        .expect("valid FireRed-OCR line detection branch kernel")
}

/// Bindings for line detection branch.
fn firered_line_detection_branch_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[LINE_DET_CH, HIDDEN_DIM]), WEIGHT_MAG);
    let proj_bias = ArrayD::from_elem(IxDyn(&[LINE_DET_CH]), 0.0f32);
    let head_w = ArrayD::from_elem(IxDyn(&[1, LINE_DET_CH]), WEIGHT_MAG);
    let head_bias = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                  // encoder_features
        TensorParamBinding::ConstantTensor(proj_w),    // det_proj_weight
        TensorParamBinding::ConstantTensor(proj_bias), // det_proj_bias
        TensorParamBinding::ConstantTensor(head_w),    // det_head_weight
        TensorParamBinding::ConstantTensor(head_bias), // det_head_bias
    ]
}

/// IBP through line detection branch: Linear -> ReLU -> Linear -> Sigmoid.
///
/// Output confidence scores must be in [0, 1] via sigmoid terminal.
#[test]
fn test_firered_line_detection_branch_ibp() {
    let def = build_firered_line_detection_branch_kernel();
    let bindings = firered_line_detection_branch_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR line detection branch");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, 1],
        "line detection branch output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR line detection branch IBP (encoder [-2,2]): bounds=[{lo_min}, {hi_max}]"
    );

    // Sigmoid output must be in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "line detection branch output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "line detection branch output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 19. Multi-head attention at 2B-scale dims CROWN
// ===========================================================================

/// CROWN bounds through 12-head self-attention at 2B-scale dimensions.
///
/// Tests CROWN linearization through the attention mechanism with scaled
/// dot-product and residual connection. The scale factor 1/sqrt(HEAD_DIM)
/// bounds the attention weights, enabling tighter CROWN propagation.
#[test]
fn test_firered_small_attention_crown() {
    let def = build_firered_small_attention_kernel();
    let bindings = firered_small_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR 12-head attention CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Residual connection with small weights keeps bounds reasonable
    assert!(
        lo_min > -100.0,
        "attention CROWN lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 20. RMSNorm -> SwiGLU -> RMSNorm sandwich IBP
// ===========================================================================

/// Build a normalization sandwich: RMSNorm -> SwiGLU FFN -> RMSNorm.
///
/// Tests normalization stability: output of SwiGLU is re-normalized.
/// In the Qwen architecture, this pattern appears between consecutive
/// encoder blocks (post-FFN norm of one block + pre-attn norm of next).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_firered_rmsnorm_swiglu_rmsnorm_sandwich_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_rmsnorm_swiglu_rmsnorm_sandwich");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Pre-FFN RMSNorm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // SwiGLU FFN
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed1, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed1, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Post-FFN RMSNorm (sandwiching)
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(ffn_out, norm2_eps, 1, norm2_w, &shape);

    b.build(out)
        .expect("valid FireRed-OCR RMSNorm-SwiGLU-RMSNorm sandwich kernel")
}

/// Bindings for RMSNorm-SwiGLU-RMSNorm sandwich.
fn firered_rmsnorm_swiglu_rmsnorm_sandwich_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
    ]
}

/// IBP bounds through RMSNorm -> SwiGLU -> RMSNorm sandwich.
///
/// The trailing RMSNorm re-normalizes SwiGLU output, constraining bound
/// widths. With unit norm weights, the output should have bounded range.
#[test]
fn test_firered_rmsnorm_swiglu_rmsnorm_sandwich_ibp() {
    let def = build_firered_rmsnorm_swiglu_rmsnorm_sandwich_kernel();
    let bindings = firered_rmsnorm_swiglu_rmsnorm_sandwich_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR RMSNorm-SwiGLU-RMSNorm sandwich");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "RMSNorm-SwiGLU-RMSNorm sandwich output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR RMSNorm-SwiGLU-RMSNorm sandwich IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 21. Patch embedding + positional encoding IBP
// ===========================================================================

/// Build patch embedding with additive positional encoding.
///
/// Conv2d patch embed -> reshape -> transpose -> add learned position bias.
/// The positional encoding is a constant tensor added to patch features,
/// testing how additive constants interact with bounded input propagation.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels in [0, 1]).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
fn build_firered_patch_embed_with_pos_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_patch_embed_with_pos");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_bias = b.add_input("patch_bias", &[HIDDEN_DIM]);
    let pos_embed = b.add_input("pos_embed", &[NUM_PATCHES, HIDDEN_DIM]);

    // Conv2d: [3, 28, 28] -> [D, 2, 2]
    let conv_out = b.add_conv2d(
        input,
        patch_w,
        Some(patch_bias),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );

    // Reshape: [D, 2, 2] -> [D, NUM_PATCHES]
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);

    // Transpose: [D, NUM_PATCHES] -> [NUM_PATCHES, D]
    let patches = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);

    // Add positional encoding
    let out = b.add_binary_add(patches, pos_embed, &[NUM_PATCHES, HIDDEN_DIM]);

    b.build(out)
        .expect("valid FireRed-OCR patch embed + pos encoding kernel")
}

/// Bindings for patch embedding with positional encoding.
fn firered_patch_embed_with_pos_bindings() -> Vec<TensorParamBinding> {
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    // Learned positional encoding: small values centered around 0
    let pos_embed = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), 0.01f32);

    vec![
        TensorParamBinding::Variable,                   // image [3, 28, 28]
        TensorParamBinding::ConstantTensor(patch_w),    // patch_weight
        TensorParamBinding::ConstantTensor(patch_bias), // patch_bias
        TensorParamBinding::ConstantTensor(pos_embed),  // pos_embed
    ]
}

/// IBP bounds through patch embedding + positional encoding.
///
/// Additive positional encoding shifts bounds by a constant offset.
/// With small pos_embed values (0.01), output bounds should be close
/// to the patch embedding output plus a small shift.
#[test]
fn test_firered_patch_embed_with_pos_ibp() {
    let def = build_firered_patch_embed_with_pos_kernel();
    let bindings = firered_patch_embed_with_pos_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR patch embed + pos encoding");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "patch embed + pos encoding output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR patch embed + pos encoding IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// 22. End-to-end CROWN: patch -> encoder -> CTC -> character probabilities
// ===========================================================================

/// CROWN bounds through the full OCR pipeline.
///
/// The same pipeline as test 8 but using CROWN linearization instead of
/// pure IBP. CROWN produces tighter bounds through the RMSNorm and SwiGLU
/// layers in the encoder. The softmax terminal still clamps to [0, 1].
#[test]
fn test_firered_ocr_pipeline_crown() {
    let def = build_firered_ocr_pipeline_kernel();
    let bindings = firered_ocr_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, VOCAB_SIZE],
        "OCR pipeline CROWN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR pipeline CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Softmax terminal: output in [0, 1] regardless of propagation method
    assert!(
        lo_min >= -1e-4,
        "OCR pipeline CROWN output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "OCR pipeline CROWN output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 23. Two-layer encoder CROWN: CROWN through 2-layer encoder stack
// ===========================================================================

/// CROWN bounds through the 2-layer encoder stack.
///
/// Tests CROWN linearization through repeated RMSNorm layers across two
/// stacked encoder blocks. CROWN should produce tighter bounds than IBP,
/// especially through the RMSNorm and multiplicative SwiGLU interactions.
#[test]
fn test_firered_two_layer_encoder_crown() {
    let def = build_firered_two_layer_encoder_kernel();
    let bindings = firered_two_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR two-layer encoder CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 24. Two-layer encoder -> CTC head IBP
// ===========================================================================

/// Build a 2-encoder-layer -> CTC head pipeline.
///
/// Reuses the 2-layer encoder followed by a CTC head (Linear + Softmax).
/// This is a lighter-weight end-to-end pipeline than test 16 (which also
/// includes patch embedding), testing deep composition ending in probability
/// bounds directly from hidden-state inputs.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, hidden states).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (character probabilities in [0, 1]).
fn build_firered_two_layer_encoder_ctc_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_two_layer_encoder_ctc");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // --- Block 1 ---
    let b1_norm1_eps = b.add_input("b1_norm1_eps", &[1]);
    let b1_norm1_w = b.add_input("b1_norm1_weight", &[HIDDEN_DIM]);
    let b1_normed1 = b.add_rms_norm(input, b1_norm1_eps, 1, b1_norm1_w, &shape);

    let b1_q_w = b.add_input("b1_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_k_w = b.add_input("b1_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_v_w = b.add_input("b1_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b1_out_w = b.add_input("b1_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let b1_q = b.add_linear(b1_normed1, b1_q_w, None, &shape);
    let b1_k = b.add_linear(b1_normed1, b1_k_w, None, &shape);
    let b1_v = b.add_linear(b1_normed1, b1_v_w, None, &shape);
    let b1_attn = b.add_attention(
        b1_q,
        b1_k,
        b1_v,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );
    let b1_attn_out = b.add_linear(b1_attn, b1_out_w, None, &shape);
    let b1_res1 = b.add_binary_add(input, b1_attn_out, &shape);

    let b1_norm2_eps = b.add_input("b1_norm2_eps", &[1]);
    let b1_norm2_w = b.add_input("b1_norm2_weight", &[HIDDEN_DIM]);
    let b1_normed2 = b.add_rms_norm(b1_res1, b1_norm2_eps, 1, b1_norm2_w, &shape);

    let b1_gate_w = b.add_input("b1_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b1_up_w = b.add_input("b1_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b1_down_w = b.add_input("b1_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let b1_gate = b.add_linear(b1_normed2, b1_gate_w, None, &ffn_shape);
    let b1_gate_sig = b.add_sigmoid(b1_gate, &ffn_shape);
    let b1_gate_act = b.add_binary_mul(b1_gate, b1_gate_sig, &ffn_shape);
    let b1_up = b.add_linear(b1_normed2, b1_up_w, None, &ffn_shape);
    let b1_hidden = b.add_binary_mul(b1_gate_act, b1_up, &ffn_shape);
    let b1_ffn_out = b.add_linear(b1_hidden, b1_down_w, None, &shape);
    let b1_res2 = b.add_binary_add(b1_res1, b1_ffn_out, &shape);

    // --- Block 2 ---
    let b2_norm1_eps = b.add_input("b2_norm1_eps", &[1]);
    let b2_norm1_w = b.add_input("b2_norm1_weight", &[HIDDEN_DIM]);
    let b2_normed1 = b.add_rms_norm(b1_res2, b2_norm1_eps, 1, b2_norm1_w, &shape);

    let b2_q_w = b.add_input("b2_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_k_w = b.add_input("b2_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_v_w = b.add_input("b2_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let b2_out_w = b.add_input("b2_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let b2_q = b.add_linear(b2_normed1, b2_q_w, None, &shape);
    let b2_k = b.add_linear(b2_normed1, b2_k_w, None, &shape);
    let b2_v = b.add_linear(b2_normed1, b2_v_w, None, &shape);
    let b2_attn = b.add_attention(
        b2_q,
        b2_k,
        b2_v,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );
    let b2_attn_out = b.add_linear(b2_attn, b2_out_w, None, &shape);
    let b2_res1 = b.add_binary_add(b1_res2, b2_attn_out, &shape);

    let b2_norm2_eps = b.add_input("b2_norm2_eps", &[1]);
    let b2_norm2_w = b.add_input("b2_norm2_weight", &[HIDDEN_DIM]);
    let b2_normed2 = b.add_rms_norm(b2_res1, b2_norm2_eps, 1, b2_norm2_w, &shape);

    let b2_gate_w = b.add_input("b2_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b2_up_w = b.add_input("b2_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let b2_down_w = b.add_input("b2_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let b2_gate = b.add_linear(b2_normed2, b2_gate_w, None, &ffn_shape);
    let b2_gate_sig = b.add_sigmoid(b2_gate, &ffn_shape);
    let b2_gate_act = b.add_binary_mul(b2_gate, b2_gate_sig, &ffn_shape);
    let b2_up = b.add_linear(b2_normed2, b2_up_w, None, &ffn_shape);
    let b2_hidden = b.add_binary_mul(b2_gate_act, b2_up, &ffn_shape);
    let b2_ffn_out = b.add_linear(b2_hidden, b2_down_w, None, &shape);
    let enc_out = b.add_binary_add(b2_res1, b2_ffn_out, &shape);

    // --- CTC head + softmax ---
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);

    let logits = b.add_linear(enc_out, ctc_w, Some(ctc_bias), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid FireRed-OCR two-layer encoder + CTC kernel")
}

/// Bindings for 2-encoder-layer -> CTC head pipeline.
fn firered_two_layer_encoder_ctc_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _block in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gate_weight
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // up_weight
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // down_weight
    }

    bindings.push(TensorParamBinding::ConstantTensor(ctc_w)); // ctc_weight
    bindings.push(TensorParamBinding::ConstantTensor(ctc_bias)); // ctc_bias

    bindings
}

/// IBP through 2-encoder-layer -> CTC head: hidden -> char probabilities.
///
/// Tests deep composition ending in probability bounds. Output must be
/// in [0, 1] via softmax terminal. Unlike test 16, starts from hidden
/// states (no patch embedding) to isolate encoder+CTC composition.
#[test]
fn test_firered_two_layer_encoder_ctc_ibp() {
    let def = build_firered_two_layer_encoder_ctc_kernel();
    let bindings = firered_two_layer_encoder_ctc_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR two-layer encoder + CTC");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "two-layer encoder + CTC output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR two-layer encoder + CTC IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]"
    );

    // Softmax terminal: output in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "two-layer encoder + CTC output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "two-layer encoder + CTC output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 25. Multi-head CTC: parallel CTC heads with fusion (IBP)
// ===========================================================================

/// Build two parallel CTC heads on shared encoder features.
///
/// Shared encoder hidden -> two independent Linear -> Softmax CTC heads.
/// Tests that parallel branches preserve independent probability bounds.
fn build_firered_multi_head_ctc_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_multi_head_ctc");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let vocab_shape = [SEQ_LEN, VOCAB_SIZE];

    // CTC head A
    let ctc_a_w = b.add_input("ctc_a_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_a_b = b.add_input("ctc_a_bias", &[VOCAB_SIZE]);
    let logits_a = b.add_linear(input, ctc_a_w, Some(ctc_a_b), &vocab_shape);
    let probs_a = b.add_softmax(logits_a, -1, &vocab_shape);

    // CTC head B
    let ctc_b_w = b.add_input("ctc_b_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_b_b = b.add_input("ctc_b_bias", &[VOCAB_SIZE]);
    let logits_b = b.add_linear(input, ctc_b_w, Some(ctc_b_b), &vocab_shape);
    let probs_b = b.add_softmax(logits_b, -1, &vocab_shape);

    // Average of two heads
    let sum = b.add_binary_add(probs_a, probs_b, &vocab_shape);
    let half = b.add_input("half_scalar", &[1]);
    let half_broadcast = b.add_broadcast(half, &vocab_shape);
    let out = b.add_binary_mul(sum, half_broadcast, &vocab_shape);

    b.build(out)
        .expect("valid FireRed-OCR multi-head CTC kernel")
}

fn firered_multi_head_ctc_bindings() -> Vec<TensorParamBinding> {
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_b = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                      // hidden
        TensorParamBinding::ConstantTensor(ctc_w.clone()), // ctc_a_weight
        TensorParamBinding::ConstantTensor(ctc_b.clone()), // ctc_a_bias
        TensorParamBinding::ConstantTensor(ctc_w),         // ctc_b_weight
        TensorParamBinding::ConstantTensor(ctc_b),         // ctc_b_bias
        TensorParamBinding::ConstantScalar(0.5),           // half_scalar
    ]
}

/// IBP through parallel CTC heads: averaged character probabilities in [0, 1].
#[test]
fn test_firered_multi_head_ctc_ibp() {
    let def = build_firered_multi_head_ctc_kernel();
    let bindings = firered_multi_head_ctc_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR multi-head CTC");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "multi-head CTC output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR multi-head CTC IBP: bounds=[{lo_min}, {hi_max}]");

    // Averaged softmax outputs should remain in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "multi-head CTC lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "multi-head CTC upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 26. Encoder 4-layer deep stack (IBP)
// ===========================================================================

/// Build a 4-layer encoder stack.
///
/// Tests bound propagation through 4 chained RMSNorm -> Attention -> Residual
/// -> SwiGLU -> Residual blocks, verifying bounds remain finite at depth.
fn build_firered_four_layer_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_four_layer_encoder");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut prev = input;
    for i in 0..4 {
        let prefix = format!("b{i}");

        let norm1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let norm1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(prev, norm1_eps, 1, norm1_w, &shape);

        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, q_w, None, &shape);
        let k = b.add_linear(normed1, k_w, None, &shape);
        let v = b.add_linear(normed1, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(prev, attn_out, &shape);

        let norm2_eps = b.add_input(&format!("{prefix}_norm2_eps"), &[1]);
        let norm2_w = b.add_input(&format!("{prefix}_norm2_weight"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

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
        .expect("valid FireRed-OCR four-layer encoder kernel")
}

fn firered_four_layer_encoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
    for _block in 0..4 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gate_weight
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // up_weight
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // down_weight
    }
    bindings
}

/// IBP through 4-layer encoder stack: bounds must remain finite.
#[test]
fn test_firered_four_layer_encoder_ibp() {
    let def = build_firered_four_layer_encoder_kernel();
    let bindings = firered_four_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR four-layer encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "four-layer encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR four-layer encoder IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 27. Encoder 8-layer deep stack for 2B depth (IBP)
// ===========================================================================

/// Build an 8-layer encoder stack representative of Qwen3-VL-2B depth.
///
/// Qwen3-VL-2B has 24 layers; 8 layers tests deep bound propagation
/// without excessive test runtime. Verifies bounds remain finite at depth.
fn build_firered_eight_layer_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_eight_layer_encoder");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut prev = input;
    for i in 0..8 {
        let prefix = format!("b{i}");

        let norm1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let norm1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(prev, norm1_eps, 1, norm1_w, &shape);

        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, q_w, None, &shape);
        let k = b.add_linear(normed1, k_w, None, &shape);
        let v = b.add_linear(normed1, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(prev, attn_out, &shape);

        let norm2_eps = b.add_input(&format!("{prefix}_norm2_eps"), &[1]);
        let norm2_w = b.add_input(&format!("{prefix}_norm2_weight"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

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
        .expect("valid FireRed-OCR eight-layer encoder kernel")
}

fn firered_eight_layer_encoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
    for _block in 0..8 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone()));
    }
    bindings
}

/// IBP through 8-layer encoder stack: finite bounds at 2B-representative depth.
#[test]
fn test_firered_eight_layer_encoder_ibp() {
    let def = build_firered_eight_layer_encoder_kernel();
    let bindings = firered_eight_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR eight-layer encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "eight-layer encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR eight-layer encoder IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite at 8 layers");
    assert!(hi_max.is_finite(), "upper bound must be finite at 8 layers");
}

// ===========================================================================
// 28. Residual accumulation bounds (IBP + CROWN)
// ===========================================================================

/// Build a residual accumulation test: input -> 3 residual adds with linear.
///
/// Tests that residual connections accumulate bounds predictably:
/// x -> x + W1 x -> (x + W1 x) + W2 (x + W1 x) -> ...
/// Each residual adds more width; verify bounds stay finite and ordered.
fn build_firered_residual_accumulation_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_residual_accumulation");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Residual 1: x + Linear(x)
    let w1 = b.add_input("w1", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj1 = b.add_linear(input, w1, None, &shape);
    let res1 = b.add_binary_add(input, proj1, &shape);

    // Residual 2: res1 + Linear(res1)
    let w2 = b.add_input("w2", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj2 = b.add_linear(res1, w2, None, &shape);
    let res2 = b.add_binary_add(res1, proj2, &shape);

    // Residual 3: res2 + Linear(res2)
    let w3 = b.add_input("w3", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj3 = b.add_linear(res2, w3, None, &shape);
    let out = b.add_binary_add(res2, proj3, &shape);

    b.build(out).expect("valid residual accumulation kernel")
}

fn firered_residual_accumulation_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                  // hidden
        TensorParamBinding::ConstantTensor(w.clone()), // w1
        TensorParamBinding::ConstantTensor(w.clone()), // w2
        TensorParamBinding::ConstantTensor(w),         // w3
    ]
}

/// IBP through 3 chained residual additions: bounds widen monotonically.
#[test]
fn test_firered_residual_accumulation_ibp() {
    let def = build_firered_residual_accumulation_kernel();
    let bindings = firered_residual_accumulation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through residual accumulation");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "residual accumulation output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR residual accumulation IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");

    // Residual accumulation widens bounds beyond input [-1, 1]
    let width = hi_max - lo_min;
    assert!(
        width >= 2.0,
        "residual accumulation should widen bounds beyond input range, got width {width}"
    );
}

/// CROWN through residual accumulation: should be tighter than IBP.
#[test]
fn test_firered_residual_accumulation_crown() {
    let def = build_firered_residual_accumulation_kernel();
    let bindings = firered_residual_accumulation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, crown_output, _fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    let (lo_min, hi_max) = bounds_min_max(&crown_output);
    eprintln!(
        "FireRed-OCR residual accumulation CROWN (method={method:?}): bounds=[{lo_min}, {hi_max}]"
    );
    assert!(lo_min.is_finite(), "CROWN lower bound must be finite");
    assert!(hi_max.is_finite(), "CROWN upper bound must be finite");
}

// ===========================================================================
// 29. RMSNorm + SwiGLU composition (CROWN)
// ===========================================================================

/// Build RMSNorm -> SwiGLU sub-block for CROWN linearization.
///
/// Tests CROWN's ability to linearize through RMSNorm (IbpValidated mode)
/// followed by SwiGLU multiplicative gating (McCormick envelopes).
fn build_firered_rmsnorm_swiglu_crown_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_rmsnorm_swiglu_crown");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // RMSNorm
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, norm_eps, 1, norm_w, &shape);

    // SwiGLU FFN
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &shape);

    b.build(out).expect("valid RMSNorm + SwiGLU kernel")
}

fn firered_rmsnorm_swiglu_crown_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantScalar(1e-5),   // norm_eps
        TensorParamBinding::ConstantTensor(norm_w), // norm_weight
        TensorParamBinding::ConstantTensor(gate_w), // gate_weight
        TensorParamBinding::ConstantTensor(up_w),   // up_weight
        TensorParamBinding::ConstantTensor(down_w), // down_weight
    ]
}

/// CROWN through RMSNorm -> SwiGLU: tighter bounds via linearization.
#[test]
fn test_firered_rmsnorm_swiglu_crown() {
    let def = build_firered_rmsnorm_swiglu_crown_kernel();
    let bindings = firered_rmsnorm_swiglu_crown_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, crown_output, _fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        crown_output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "RMSNorm + SwiGLU CROWN output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&crown_output);
    eprintln!(
        "FireRed-OCR RMSNorm + SwiGLU CROWN (method={method:?}): bounds=[{lo_min}, {hi_max}]"
    );
    assert!(lo_min.is_finite(), "CROWN lower bound must be finite");
    assert!(hi_max.is_finite(), "CROWN upper bound must be finite");
}

// ===========================================================================
// 30. Full encoder -> CTC -> softmax 4 layers (IBP)
// ===========================================================================

/// Build 4-layer encoder -> CTC head -> softmax pipeline.
///
/// Deeper composition than test 16 (2 layers). Tests that softmax terminal
/// still produces [0, 1] bounds after 4 encoder layers of bound widening.
fn build_firered_four_layer_encoder_ctc_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_four_layer_encoder_ctc");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut prev = input;
    for i in 0..4 {
        let prefix = format!("b{i}");

        let norm1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let norm1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(prev, norm1_eps, 1, norm1_w, &shape);

        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, q_w, None, &shape);
        let k = b.add_linear(normed1, k_w, None, &shape);
        let v = b.add_linear(normed1, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(prev, attn_out, &shape);

        let norm2_eps = b.add_input(&format!("{prefix}_norm2_eps"), &[1]);
        let norm2_w = b.add_input(&format!("{prefix}_norm2_weight"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

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

    // CTC head + softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(prev, ctc_w, Some(ctc_bias), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid FireRed-OCR four-layer encoder + CTC kernel")
}

fn firered_four_layer_encoder_ctc_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
    for _block in 0..4 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone()));
    }
    bindings.push(TensorParamBinding::ConstantTensor(ctc_w));
    bindings.push(TensorParamBinding::ConstantTensor(ctc_bias));
    bindings
}

/// IBP through 4-layer encoder + CTC: character probabilities in [0, 1].
#[test]
fn test_firered_four_layer_encoder_ctc_ibp() {
    let def = build_firered_four_layer_encoder_ctc_kernel();
    let bindings = firered_four_layer_encoder_ctc_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through FireRed-OCR four-layer encoder + CTC");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "four-layer encoder + CTC output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR four-layer encoder + CTC IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(
        lo_min >= -1e-4,
        "four-layer encoder + CTC output lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "four-layer encoder + CTC output upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 31. Cross-attention encoder <-> line positions (IBP)
// ===========================================================================

/// Line position feature dimension for cross-attention.
const LINE_POS_DIM: usize = 16;
/// Number of detected line positions.
const NUM_LINES: usize = 8;

/// Build cross-attention: encoder features attend to line position features.
///
/// Encoder hidden states (Q) attend to line position embeddings (K, V).
/// This models how FireRed-OCR uses spatial line position information
/// to focus the encoder on text line regions.
fn build_firered_cross_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_cross_attention");

    let enc_input = b.add_input("encoder_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let line_input = b.add_input("line_positions", &[NUM_LINES, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Project Q from encoder, K/V from line positions
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(enc_input, q_w, None, &shape);
    let kv_shape = [NUM_LINES, HIDDEN_DIM];
    let k = b.add_linear(line_input, k_w, None, &kv_shape);
    let v = b.add_linear(line_input, v_w, None, &kv_shape);

    // Cross-attention: Q from encoder, K/V from line positions
    // Use matmul-based attention: Q @ K^T -> softmax -> @ V
    // Q: [SEQ_LEN, HIDDEN_DIM], K^T: [HIDDEN_DIM, NUM_LINES]
    let kt = b.add_transpose(k, &[1, 0], &[HIDDEN_DIM, NUM_LINES]);
    let scores = b.add_matmul(q, kt, false, None, &[SEQ_LEN, NUM_LINES]);

    // Scale by 1/sqrt(HEAD_DIM)
    let scale_val = b.add_input("scale", &[1]);
    let scale_broadcast = b.add_broadcast(scale_val, &[SEQ_LEN, NUM_LINES]);
    let scaled_scores = b.add_binary_mul(scores, scale_broadcast, &[SEQ_LEN, NUM_LINES]);

    let attn_weights = b.add_softmax(scaled_scores, -1, &[SEQ_LEN, NUM_LINES]);

    // attn_weights @ V: [SEQ_LEN, NUM_LINES] @ [NUM_LINES, HIDDEN_DIM]
    let context = b.add_matmul(attn_weights, v, false, None, &shape);
    let out = b.add_linear(context, out_w, None, &shape);

    b.build(out).expect("valid cross-attention kernel")
}

fn firered_cross_attention_bindings() -> Vec<TensorParamBinding> {
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    vec![
        TensorParamBinding::Variable,                       // encoder_hidden
        TensorParamBinding::Variable,                       // line_positions
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(qkvo_w),         // out_weight
        TensorParamBinding::ConstantScalar(scale),          // scale
    ]
}

/// IBP through cross-attention between encoder and line positions.
#[test]
fn test_firered_cross_attention_ibp() {
    let def = build_firered_cross_attention_kernel();
    let bindings = firered_cross_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Two variable inputs concatenated: encoder_hidden and line_positions
    // Variable inputs: [SEQ_LEN*HIDDEN_DIM + NUM_LINES*HIDDEN_DIM]
    let total_elems = SEQ_LEN * HIDDEN_DIM + NUM_LINES * HIDDEN_DIM;
    let input = uniform_bounds(&[total_elems], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attention");

    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR cross-attention IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 32. Token embedding -> encoder -> CTC (CROWN)
// ===========================================================================

/// Build token embedding -> single encoder layer -> CTC pipeline.
///
/// Tests CROWN linearization through the full sequence:
/// embedding lookup -> RMSNorm -> attention -> SwiGLU -> CTC softmax.
fn build_firered_embedding_encoder_ctc_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_embedding_encoder_ctc");

    // Token indices as Variable input (embedded via lookup)
    let token_ids = b.add_input("token_ids", &[SEQ_LEN]);
    let embed_w = b.add_input("embed_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let embedded = b.add_embedding(token_ids, embed_w, &[SEQ_LEN, HIDDEN_DIM]);

    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Encoder layer
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
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
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
    let enc_out = b.add_binary_add(res1, ffn_out, &shape);

    // CTC head
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(enc_out, ctc_w, Some(ctc_bias), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out)
        .expect("valid embedding -> encoder -> CTC kernel")
}

fn firered_embedding_encoder_ctc_bindings() -> Vec<TensorParamBinding> {
    let embed_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                       // token_ids
        TensorParamBinding::ConstantTensor(embed_w),        // embed_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(qkvo_w),         // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
        TensorParamBinding::ConstantTensor(ctc_w),          // ctc_weight
        TensorParamBinding::ConstantTensor(ctc_bias),       // ctc_bias
    ]
}

/// CROWN through embedding -> encoder -> CTC pipeline.
#[test]
fn test_firered_embedding_encoder_ctc_crown() {
    let def = build_firered_embedding_encoder_ctc_kernel();
    let bindings = firered_embedding_encoder_ctc_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Token ID indices as variable: bounded in [0, VOCAB_SIZE-1] range
    let input = uniform_bounds(&[SEQ_LEN], (VOCAB_SIZE - 1) as f32);

    let (method, crown_output, _fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        crown_output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "embedding -> encoder -> CTC CROWN output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&crown_output);
    eprintln!(
        "FireRed-OCR embedding -> encoder -> CTC CROWN (method={method:?}): \
         bounds=[{lo_min}, {hi_max}]"
    );

    // Softmax terminal: output in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "embedding -> encoder -> CTC CROWN lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "embedding -> encoder -> CTC CROWN upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 33. Batch B>1 bounds (IBP)
// ===========================================================================

/// Batch size for multi-batch tests.
const BATCH_SIZE: usize = 2;

/// Build encoder layer with batch dimension: [B, SEQ_LEN, HIDDEN_DIM].
///
/// Tests that IBP bounds propagation handles the batch dimension correctly.
/// Batch dimension is treated as an outer dimension that does not interact
/// with the per-sequence computation.
fn build_firered_batched_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_batched_encoder");

    let input = b.add_input("hidden", &[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM]);
    let shape = [BATCH_SIZE, SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [BATCH_SIZE, SEQ_LEN, FFN_DIM];

    // RMSNorm (normalizes over last dim — axis 2 of [B, SEQ_LEN, HIDDEN_DIM])
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, norm_eps, 2, norm_w, &shape);

    // SwiGLU FFN
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Residual
    let out = b.add_binary_add(input, ffn_out, &shape);

    b.build(out).expect("valid batched encoder kernel")
}

fn firered_batched_encoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantScalar(1e-5),   // norm_eps
        TensorParamBinding::ConstantTensor(norm_w), // norm_weight
        TensorParamBinding::ConstantTensor(gate_w), // gate_weight
        TensorParamBinding::ConstantTensor(up_w),   // up_weight
        TensorParamBinding::ConstantTensor(down_w), // down_weight
    ]
}

/// IBP through batched encoder: batch dimension preserved in output.
#[test]
fn test_firered_batched_encoder_ibp() {
    let def = build_firered_batched_encoder_kernel();
    let bindings = firered_batched_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through batched encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[BATCH_SIZE, SEQ_LEN, HIDDEN_DIM],
        "batched encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR batched encoder IBP (B={BATCH_SIZE}): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "batched lower bound must be finite");
    assert!(hi_max.is_finite(), "batched upper bound must be finite");
}

// ===========================================================================
// 34. Padding invariance (IBP)
// ===========================================================================

/// Padded sequence length (SEQ_LEN + padding).
const PADDED_SEQ_LEN: usize = SEQ_LEN + 4;

/// Build encoder with zero-padded input sequence.
///
/// Tests that zero-padding tokens do not invalidate bounds for the
/// non-padded positions. The padded region uses bounded [0, 0] (exact zero)
/// while the real tokens use [-1, 1].
fn build_firered_padded_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_padded_encoder");

    let input = b.add_input("hidden", &[PADDED_SEQ_LEN, HIDDEN_DIM]);
    let shape = [PADDED_SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [PADDED_SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // RMSNorm -> Attention -> Residual -> SwiGLU -> Residual
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

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
    let out = b.add_binary_add(res1, ffn_out, &shape);

    b.build(out).expect("valid padded encoder kernel")
}

fn firered_padded_encoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(qkvo_w),         // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
    ]
}

/// IBP through encoder with padded input: bounds remain valid at longer sequence.
///
/// Compares padded (SEQ_LEN+4) vs. non-padded (SEQ_LEN) encoder to verify
/// that padding does not break bound validity. Both should produce finite,
/// well-ordered bounds.
#[test]
fn test_firered_padded_encoder_ibp() {
    let def = build_firered_padded_encoder_kernel();
    let bindings = firered_padded_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Build input with real tokens [-1, 1] and padding [0, 0]:
    // For IBP, we use a single BoundedTensor covering the full padded sequence.
    // The padding positions have narrower bounds (closer to zero) simulating
    // zero-padding, while real positions use full [-1, 1].
    let n = PADDED_SEQ_LEN * HIDDEN_DIM;
    let mut lower = vec![-1.0f32; n];
    let mut upper = vec![1.0f32; n];

    // Zero out padding region (last 4 positions)
    for pos in SEQ_LEN..PADDED_SEQ_LEN {
        for d in 0..HIDDEN_DIM {
            let idx = pos * HIDDEN_DIM + d;
            lower[idx] = 0.0;
            upper[idx] = 0.0;
        }
    }

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[PADDED_SEQ_LEN, HIDDEN_DIM]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[PADDED_SEQ_LEN, HIDDEN_DIM]), upper).unwrap(),
    )
    .expect("valid padded input bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through padded encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[PADDED_SEQ_LEN, HIDDEN_DIM],
        "padded encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR padded encoder IBP (pad=4): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "padded lower bound must be finite");
    assert!(hi_max.is_finite(), "padded upper bound must be finite");

    // Also run the unpadded version and verify both are finite
    let unpadded_def = build_firered_encoder_layer_kernel();
    let unpadded_bindings = firered_encoder_layer_bindings();
    let unpadded_graph =
        tensor_kernel_to_graph(&unpadded_def, &unpadded_bindings).expect("graph translation");
    let unpadded_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let unpadded_output = unpadded_graph
        .propagate_ibp(&unpadded_input)
        .expect("IBP through unpadded encoder");
    assert_bounds_valid(&unpadded_output);

    let (unpadded_lo, unpadded_hi) = bounds_min_max(&unpadded_output);
    eprintln!("FireRed-OCR unpadded encoder IBP: bounds=[{unpadded_lo}, {unpadded_hi}]");
}

// ===========================================================================
// 35. Deep 8-layer encoder CROWN: CROWN linearization at 2B-representative depth
// ===========================================================================

/// CROWN bounds through the full 8-layer encoder stack.
///
/// Tests CROWN linearization through 8 stacked RMSNorm + attention + SwiGLU
/// blocks. At this depth, IBP bounds can blow up significantly; CROWN should
/// produce tighter bounds through linearization of the repeated nonlinearities.
#[test]
fn test_firered_eight_layer_encoder_crown() {
    let def = build_firered_eight_layer_encoder_kernel();
    let bindings = firered_eight_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR eight-layer encoder CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(
        lo_min.is_finite(),
        "lower bound must be finite at 8 layers (CROWN)"
    );
    assert!(
        hi_max.is_finite(),
        "upper bound must be finite at 8 layers (CROWN)"
    );
}

// ===========================================================================
// 36. Large 24-head attention at 2048 dims (scaled down) IBP
// ===========================================================================

/// Hidden dimension for 24-head attention test (scaled down from 2048).
const LARGE_HIDDEN_DIM: usize = 72;
/// Number of heads for the large attention test.
const LARGE_NUM_HEADS: usize = 24;
/// Head dimension for 24-head: 72 / 24 = 3.
const LARGE_HEAD_DIM: usize = LARGE_HIDDEN_DIM / LARGE_NUM_HEADS;

/// Build a 24-head attention block at larger hidden dimension.
///
/// Tests that bounds propagate correctly through attention with more heads
/// and wider hidden dimension, representative of Qwen3-VL-2B scale
/// (2048 dims, 24 heads in production; scaled down to 72 dims here).
///
/// Input: `[SEQ_LEN, LARGE_HIDDEN_DIM]`.
/// Output: `[SEQ_LEN, LARGE_HIDDEN_DIM]`.
fn build_firered_large_24head_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_large_24head_attention");

    let input = b.add_input("hidden", &[SEQ_LEN, LARGE_HIDDEN_DIM]);
    let shape = [SEQ_LEN, LARGE_HIDDEN_DIM];
    let scale = 1.0 / (LARGE_HEAD_DIM as f32).sqrt();

    let q_w = b.add_input("q_weight", &[LARGE_HIDDEN_DIM, LARGE_HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[LARGE_HIDDEN_DIM, LARGE_HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[LARGE_HIDDEN_DIM, LARGE_HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[LARGE_HIDDEN_DIM, LARGE_HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(input, attn_out, &shape);

    b.build(out)
        .expect("valid FireRed-OCR large 24-head attention kernel")
}

fn firered_large_24head_attention_bindings() -> Vec<TensorParamBinding> {
    let qkvo_w = ArrayD::from_elem(IxDyn(&[LARGE_HIDDEN_DIM, LARGE_HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(qkvo_w),         // out_weight
    ]
}

/// IBP through 24-head attention at 72 dims.
#[test]
fn test_firered_large_24head_attention_ibp() {
    let def = build_firered_large_24head_attention_kernel();
    let bindings = firered_large_24head_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, LARGE_HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 24-head attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, LARGE_HIDDEN_DIM],
        "24-head attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR 24-head attention IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "24-head lower must be finite");
    assert!(hi_max.is_finite(), "24-head upper must be finite");
}

// ===========================================================================
// 37. SwiGLU FFN at 5632 intermediate dims (scaled down) IBP
// ===========================================================================

/// Large FFN intermediate dimension (scaled down from 5632 for tractability).
const LARGE_FFN_DIM: usize = 128;

/// Build a SwiGLU FFN at larger intermediate dimension.
///
/// Tests that SwiGLU gating bounds propagate at wider FFN dimensions
/// representative of the 2B-scale model (5632 -> 128 scaled down).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]`.
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_firered_large_swiglu_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_large_swiglu_ffn");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, LARGE_FFN_DIM];

    let gate_w = b.add_input("gate_weight", &[LARGE_FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[LARGE_FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, LARGE_FFN_DIM]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &shape);

    b.build(out)
        .expect("valid FireRed-OCR large SwiGLU FFN kernel")
}

fn firered_large_swiglu_ffn_bindings() -> Vec<TensorParamBinding> {
    let gate_w = ArrayD::from_elem(IxDyn(&[LARGE_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[LARGE_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, LARGE_FFN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantTensor(gate_w), // gate_weight
        TensorParamBinding::ConstantTensor(up_w),   // up_weight
        TensorParamBinding::ConstantTensor(down_w), // down_weight
    ]
}

/// IBP through SwiGLU FFN at large intermediate dimension.
#[test]
fn test_firered_large_swiglu_ffn_ibp() {
    let def = build_firered_large_swiglu_ffn_kernel();
    let bindings = firered_large_swiglu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through large SwiGLU FFN");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "large SwiGLU FFN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR large SwiGLU FFN IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "large SwiGLU FFN lower must be finite");
    assert!(hi_max.is_finite(), "large SwiGLU FFN upper must be finite");
}

// ===========================================================================
// 38. Residual accumulation through 8+ layers IBP: bound widening test
// ===========================================================================

/// Build 8-deep residual chain (linear + add) without attention/FFN.
///
/// Tests pure residual accumulation: x -> x + W1x -> (x+W1x) + W2(x+W1x) -> ...
/// for 8 layers. This isolates the residual contribution to bound widening
/// from attention/FFN effects.
fn build_firered_deep_residual_chain_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_deep_residual_chain");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut prev = input;
    for i in 0..8 {
        let w = b.add_input(&format!("w{i}"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let proj = b.add_linear(prev, w, None, &shape);
        prev = b.add_binary_add(prev, proj, &shape);
    }

    b.build(prev)
        .expect("valid FireRed-OCR deep residual chain kernel")
}

fn firered_deep_residual_chain_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
    for _ in 0..8 {
        bindings.push(TensorParamBinding::ConstantTensor(w.clone()));
    }
    bindings
}

/// IBP through 8-deep residual chain.
#[test]
fn test_firered_deep_residual_chain_ibp() {
    let def = build_firered_deep_residual_chain_kernel();
    let bindings = firered_deep_residual_chain_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through deep residual chain");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "deep residual chain output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR deep residual chain IBP (8 layers): bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "deep residual lower must be finite");
    assert!(hi_max.is_finite(), "deep residual upper must be finite");

    // Bounds should widen compared to single-layer. Check not degenerate.
    assert!(
        hi_max > 0.0,
        "deep residual upper must be positive (non-degenerate)"
    );
}

// ===========================================================================
// 39. RMSNorm stability at large dimensions (IBP)
// ===========================================================================

/// Build RMSNorm at LARGE_HIDDEN_DIM (72) to test stability at wider dims.
///
/// RMSNorm normalizes by 1/sqrt(mean(x^2) + eps). At larger dimensions, the
/// mean is over more elements, which can affect bound propagation stability.
fn build_firered_rmsnorm_large_dim_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_rmsnorm_large_dim");

    let input = b.add_input("hidden", &[SEQ_LEN, LARGE_HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[LARGE_HIDDEN_DIM]);

    let out = b.add_rms_norm(input, eps, 1, weight, &[SEQ_LEN, LARGE_HIDDEN_DIM]);

    b.build(out)
        .expect("valid RMSNorm at large dimension kernel")
}

fn firered_rmsnorm_large_dim_bindings() -> Vec<TensorParamBinding> {
    let weight = ArrayD::from_elem(IxDyn(&[LARGE_HIDDEN_DIM]), 1.0f32);
    vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantScalar(1e-5),   // eps
        TensorParamBinding::ConstantTensor(weight), // weight
    ]
}

/// IBP through RMSNorm at 72 dims: bounds must remain stable.
#[test]
fn test_firered_rmsnorm_large_dim_ibp() {
    let def = build_firered_rmsnorm_large_dim_kernel();
    let bindings = firered_rmsnorm_large_dim_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, LARGE_HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through RMSNorm at large dim");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, LARGE_HIDDEN_DIM],
        "large-dim RMSNorm output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR RMSNorm (72 dims) IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "large-dim RMSNorm lower must be finite");
    assert!(hi_max.is_finite(), "large-dim RMSNorm upper must be finite");
}

// ===========================================================================
// 40. RMSNorm stability at large dimensions (CROWN)
// ===========================================================================

/// CROWN through RMSNorm at 72 dims: tighter bounds via linearization.
#[test]
fn test_firered_rmsnorm_large_dim_crown() {
    let def = build_firered_rmsnorm_large_dim_kernel();
    let bindings = firered_rmsnorm_large_dim_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, LARGE_HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, LARGE_HIDDEN_DIM],
        "large-dim RMSNorm CROWN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR RMSNorm (72 dims) CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(
        lo_min.is_finite(),
        "large-dim RMSNorm CROWN lower must be finite"
    );
    assert!(
        hi_max.is_finite(),
        "large-dim RMSNorm CROWN upper must be finite"
    );
}

// ===========================================================================
// 41. Full CTC pipeline: encoder -> linear -> softmax -> argmax (IBP)
// ===========================================================================

/// Build a full CTC decoding pipeline: encoder layer -> CTC projection ->
/// softmax -> log_softmax for decoding confidence.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]`.
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (log-probabilities).
fn build_firered_full_ctc_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_full_ctc_pipeline");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let vocab_shape = [SEQ_LEN, VOCAB_SIZE];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Encoder layer
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

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
    let enc_out = b.add_binary_add(res1, ffn_out, &shape);

    // CTC head: linear -> softmax -> log_softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(enc_out, ctc_w, Some(ctc_bias), &vocab_shape);
    let probs = b.add_softmax(logits, -1, &vocab_shape);
    let log_probs = b.add_log_softmax(probs, -1, &vocab_shape);

    b.build(log_probs).expect("valid full CTC pipeline kernel")
}

fn firered_full_ctc_pipeline_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(qkvo_w),         // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
        TensorParamBinding::ConstantTensor(ctc_w),          // ctc_weight
        TensorParamBinding::ConstantTensor(ctc_bias),       // ctc_bias
    ]
}

/// IBP through full CTC pipeline with log_softmax output.
#[test]
fn test_firered_full_ctc_pipeline_ibp() {
    let def = build_firered_full_ctc_pipeline_kernel();
    let bindings = firered_full_ctc_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full CTC pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "full CTC pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR full CTC pipeline IBP: bounds=[{lo_min}, {hi_max}]");

    // log_softmax outputs are <= 0
    assert!(
        hi_max <= 1e-4,
        "log_softmax output should be <= 0, got upper {hi_max}"
    );
    assert!(lo_min.is_finite(), "full CTC pipeline lower must be finite");
}

// ===========================================================================
// 42. Blank token probability bounds (IBP): verify blank in [0, 1]
// ===========================================================================

/// Build CTC pipeline with explicit blank token extraction.
///
/// CTC blank token is typically index 0. After softmax, we verify that
/// the blank token's probability is bounded in [0, 1] for all inputs.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]`.
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (softmax probabilities).
fn build_firered_ctc_blank_extraction_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_ctc_blank_extraction");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let vocab_shape = [SEQ_LEN, VOCAB_SIZE];

    // Two-layer projection for more interesting bounds
    let proj1_w = b.add_input("proj1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let proj1_bias = b.add_input("proj1_bias", &[FFN_DIM]);
    let proj1 = b.add_linear(input, proj1_w, Some(proj1_bias), &[SEQ_LEN, FFN_DIM]);
    let activated = b.add_relu(proj1, &[SEQ_LEN, FFN_DIM]);

    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, FFN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(activated, ctc_w, Some(ctc_bias), &vocab_shape);
    let out = b.add_softmax(logits, -1, &vocab_shape);

    b.build(out).expect("valid CTC blank extraction kernel")
}

fn firered_ctc_blank_extraction_bindings() -> Vec<TensorParamBinding> {
    let proj1_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let proj1_bias = ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32);
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, FFN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);
    vec![
        TensorParamBinding::Variable,                   // hidden
        TensorParamBinding::ConstantTensor(proj1_w),    // proj1_weight
        TensorParamBinding::ConstantTensor(proj1_bias), // proj1_bias
        TensorParamBinding::ConstantTensor(ctc_w),      // ctc_weight
        TensorParamBinding::ConstantTensor(ctc_bias),   // ctc_bias
    ]
}

/// IBP through CTC blank extraction: blank probability bounded in [0, 1].
#[test]
fn test_firered_ctc_blank_extraction_ibp() {
    let def = build_firered_ctc_blank_extraction_kernel();
    let bindings = firered_ctc_blank_extraction_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through CTC blank extraction");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "CTC blank extraction output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();

    // Check blank token (index 0) probability bounds per timestep
    for t in 0..SEQ_LEN {
        let blank_lo = lo[[t, 0]];
        let blank_hi = hi[[t, 0]];
        assert!(
            blank_lo >= -1e-4,
            "blank token lower at t={t} should be >= 0, got {blank_lo}"
        );
        assert!(
            blank_hi <= 1.0 + 1e-4,
            "blank token upper at t={t} should be <= 1, got {blank_hi}"
        );
    }

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR CTC blank extraction IBP: bounds=[{lo_min}, {hi_max}]");
}

// ===========================================================================
// 43. Multi-character decoding across 65536 vocab (scaled down) IBP
// ===========================================================================

/// Large vocabulary for CTC (scaled down from 65536 for tractability).
const LARGE_VOCAB_SIZE: usize = 512;

/// Build CTC head with larger vocabulary.
///
/// Tests that softmax bounds remain in [0, 1] even with many output classes.
/// In production, FireRed-OCR uses 65536 vocab (full Unicode coverage).
fn build_firered_large_vocab_ctc_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_large_vocab_ctc");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let vocab_shape = [SEQ_LEN, LARGE_VOCAB_SIZE];

    let ctc_w = b.add_input("ctc_weight", &[LARGE_VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[LARGE_VOCAB_SIZE]);
    let logits = b.add_linear(input, ctc_w, Some(ctc_bias), &vocab_shape);
    let out = b.add_softmax(logits, -1, &vocab_shape);

    b.build(out).expect("valid large vocab CTC kernel")
}

fn firered_large_vocab_ctc_bindings() -> Vec<TensorParamBinding> {
    let ctc_w = ArrayD::from_elem(IxDyn(&[LARGE_VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[LARGE_VOCAB_SIZE]), 0.0f32);
    vec![
        TensorParamBinding::Variable,                 // hidden
        TensorParamBinding::ConstantTensor(ctc_w),    // ctc_weight
        TensorParamBinding::ConstantTensor(ctc_bias), // ctc_bias
    ]
}

/// IBP through large vocab CTC: softmax output in [0, 1].
#[test]
fn test_firered_large_vocab_ctc_ibp() {
    let def = build_firered_large_vocab_ctc_kernel();
    let bindings = firered_large_vocab_ctc_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through large vocab CTC");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, LARGE_VOCAB_SIZE],
        "large vocab CTC output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR large vocab CTC IBP (vocab={LARGE_VOCAB_SIZE}): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(
        lo_min >= -1e-4,
        "large vocab CTC lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "large vocab CTC upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 44. CTC prefix beam search monotonicity: softmax sum bounds (IBP)
// ===========================================================================

/// Build CTC softmax output and verify probability sum bounds.
///
/// CTC prefix beam search requires that softmax outputs sum to 1.0 per
/// timestep. This test verifies softmax bounds are consistent with the
/// sum-to-one property: each output is in [0, 1].
fn build_firered_ctc_sum_verification_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_ctc_sum_verification");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Simple linear + softmax
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(input, ctc_w, Some(ctc_bias), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid CTC sum verification kernel")
}

fn firered_ctc_sum_verification_bindings() -> Vec<TensorParamBinding> {
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);
    vec![
        TensorParamBinding::Variable,                 // hidden
        TensorParamBinding::ConstantTensor(ctc_w),    // ctc_weight
        TensorParamBinding::ConstantTensor(ctc_bias), // ctc_bias
    ]
}

/// IBP through CTC softmax: verify [0, 1] per-element and sum consistency.
#[test]
fn test_firered_ctc_sum_verification_ibp() {
    let def = build_firered_ctc_sum_verification_kernel();
    let bindings = firered_ctc_sum_verification_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through CTC sum verification");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "CTC sum verification output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();

    // Per-timestep: each element is in [0, 1] (softmax guarantee)
    for t in 0..SEQ_LEN {
        for v in 0..VOCAB_SIZE {
            assert!(
                lo[[t, v]] >= -1e-4,
                "CTC prob lower at [{t}, {v}] should be >= 0, got {}",
                lo[[t, v]]
            );
            assert!(
                hi[[t, v]] <= 1.0 + 1e-4,
                "CTC prob upper at [{t}, {v}] should be <= 1, got {}",
                hi[[t, v]]
            );
        }
    }

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR CTC sum verification IBP: bounds=[{lo_min}, {hi_max}]");
}

// ===========================================================================
// 45. Patch embed -> 8-layer encoder -> CTC head end-to-end (IBP)
// ===========================================================================

/// Build the full end-to-end OCR pipeline: patch embedding -> 8-layer
/// encoder -> CTC projection -> softmax.
///
/// This is the deepest end-to-end composition test, representing the
/// complete FireRed-OCR inference path from image pixels to character
/// probabilities.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels in [0, 1]).
/// Output: `[NUM_PATCHES, VOCAB_SIZE]`.
fn build_firered_full_ocr_e2e_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_full_e2e");

    // Patch embedding
    let image = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_bias = b.add_input("patch_bias", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        image,
        patch_w,
        Some(patch_bias),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);

    // 8-layer encoder
    let shape = [NUM_PATCHES, HIDDEN_DIM];
    let ffn_shape = [NUM_PATCHES, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut prev = transposed;

    for i in 0..8 {
        let prefix = format!("b{i}");

        let norm1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let norm1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(prev, norm1_eps, 1, norm1_w, &shape);

        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, q_w, None, &shape);
        let k = b.add_linear(normed1, k_w, None, &shape);
        let v = b.add_linear(normed1, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(prev, attn_out, &shape);

        let norm2_eps = b.add_input(&format!("{prefix}_norm2_eps"), &[1]);
        let norm2_w = b.add_input(&format!("{prefix}_norm2_weight"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

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

    // CTC head
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(prev, ctc_w, Some(ctc_bias), &[NUM_PATCHES, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[NUM_PATCHES, VOCAB_SIZE]);

    b.build(out).expect("valid full OCR end-to-end kernel")
}

fn firered_full_ocr_e2e_bindings() -> Vec<TensorParamBinding> {
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    let mut bindings = vec![
        TensorParamBinding::Variable,                   // image
        TensorParamBinding::ConstantTensor(patch_w),    // patch_weight
        TensorParamBinding::ConstantTensor(patch_bias), // patch_bias
    ];

    for _block in 0..8 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gate_weight
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // up_weight
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // down_weight
    }

    bindings.push(TensorParamBinding::ConstantTensor(ctc_w));
    bindings.push(TensorParamBinding::ConstantTensor(ctc_bias));
    bindings
}

/// IBP through full OCR end-to-end: image pixels -> character probabilities.
#[test]
fn test_firered_full_ocr_e2e_ibp() {
    let def = build_firered_full_ocr_e2e_kernel();
    let bindings = firered_full_ocr_e2e_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full OCR end-to-end");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, VOCAB_SIZE],
        "full OCR end-to-end output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR full e2e IBP (8 layers): bounds=[{lo_min}, {hi_max}]");

    assert!(
        lo_min >= -1e-4,
        "full OCR e2e lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "full OCR e2e upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 46. Line detection + recognition composition (IBP)
// ===========================================================================

/// Build a line detection + recognition pipeline.
///
/// Models the two-branch architecture: shared encoder features feed both
/// a line detection head (sigmoid) and a character recognition head (softmax).
/// Tests that both branches independently produce valid bounds from the same
/// encoder output.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]`.
/// Output detection: `[SEQ_LEN, LINE_DET_CH]` (sigmoid).
/// Output recognition: `[SEQ_LEN, VOCAB_SIZE]` (softmax).
///
/// For compose tests we verify the recognition branch.
fn build_firered_line_detect_recognize_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_line_detect_recognize");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Shared encoder layer
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

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
    let enc_out = b.add_binary_add(res1, ffn_out, &shape);

    // Detection branch: linear -> sigmoid
    let det_w = b.add_input("det_weight", &[LINE_DET_CH, HIDDEN_DIM]);
    let det_bias = b.add_input("det_bias", &[LINE_DET_CH]);
    let det_logits = b.add_linear(enc_out, det_w, Some(det_bias), &[SEQ_LEN, LINE_DET_CH]);
    let _det_out = b.add_sigmoid(det_logits, &[SEQ_LEN, LINE_DET_CH]);

    // Recognition branch: linear -> softmax (this is the output)
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let ctc_logits = b.add_linear(enc_out, ctc_w, Some(ctc_bias), &[SEQ_LEN, VOCAB_SIZE]);
    let recog_out = b.add_softmax(ctc_logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(recog_out)
        .expect("valid line detect + recognize kernel")
}

fn firered_line_detect_recognize_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let det_w = ArrayD::from_elem(IxDyn(&[LINE_DET_CH, HIDDEN_DIM]), WEIGHT_MAG);
    let det_bias = ArrayD::from_elem(IxDyn(&[LINE_DET_CH]), 0.0f32);
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(qkvo_w),         // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
        TensorParamBinding::ConstantTensor(det_w),          // det_weight
        TensorParamBinding::ConstantTensor(det_bias),       // det_bias
        TensorParamBinding::ConstantTensor(ctc_w),          // ctc_weight
        TensorParamBinding::ConstantTensor(ctc_bias),       // ctc_bias
    ]
}

/// IBP through line detection + recognition composition.
#[test]
fn test_firered_line_detect_recognize_ibp() {
    let def = build_firered_line_detect_recognize_kernel();
    let bindings = firered_line_detect_recognize_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through line detect + recognize");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "line detect + recognize output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR line detect + recognize IBP: bounds=[{lo_min}, {hi_max}]");

    // Recognition branch uses softmax: output in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "line detect + recognize lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "line detect + recognize upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 47. Multi-page document bounds: multiple independent patch sequences (IBP)
// ===========================================================================

/// Number of pages in a multi-page document test.
const NUM_PAGES: usize = 3;
/// Total sequence length for multi-page: NUM_PAGES * NUM_PATCHES.
const MULTI_PAGE_SEQ: usize = NUM_PAGES * NUM_PATCHES;

/// Build encoder for multi-page document: concatenated patch sequences.
///
/// In production, a multi-page document has patches from each page
/// concatenated into a single sequence. Tests that bounds remain valid
/// at longer sequences (12 patches from 3 pages vs 4 from 1 page).
fn build_firered_multi_page_encoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_multi_page_encoder");

    let input = b.add_input("hidden", &[MULTI_PAGE_SEQ, HIDDEN_DIM]);
    let shape = [MULTI_PAGE_SEQ, HIDDEN_DIM];
    let ffn_shape = [MULTI_PAGE_SEQ, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Single encoder layer over the full multi-page sequence
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

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
    let out = b.add_binary_add(res1, ffn_out, &shape);

    b.build(out).expect("valid multi-page encoder kernel")
}

fn firered_multi_page_encoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(qkvo_w),         // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
    ]
}

/// IBP through multi-page encoder: bounds at longer sequences.
#[test]
fn test_firered_multi_page_encoder_ibp() {
    let def = build_firered_multi_page_encoder_kernel();
    let bindings = firered_multi_page_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[MULTI_PAGE_SEQ, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-page encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[MULTI_PAGE_SEQ, HIDDEN_DIM],
        "multi-page encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR multi-page encoder IBP (pages={NUM_PAGES}, seq={MULTI_PAGE_SEQ}): \
         bounds=[{lo_min}, {hi_max}]"
    );
    assert!(lo_min.is_finite(), "multi-page lower must be finite");
    assert!(hi_max.is_finite(), "multi-page upper must be finite");
}

// ===========================================================================
// 48. Resolution scaling: patch embed at different resolutions (IBP)
// ===========================================================================

/// Larger image size (scaled down from 448).
const LARGE_IMG_SIZE: usize = 42;
/// Grid size at larger resolution.
const LARGE_GRID_SIZE: usize = LARGE_IMG_SIZE / PATCH_SIZE; // 3
/// Patches at larger resolution.
const LARGE_NUM_PATCHES: usize = LARGE_GRID_SIZE * LARGE_GRID_SIZE; // 9

/// Build patch embedding at larger resolution (42x42, 9 patches).
///
/// Tests that bounds scale correctly with more patches. In production,
/// FireRed-OCR handles resolutions 224, 448, and 896; here we test
/// a scaled-down version to verify resolution-dependent bounds.
fn build_firered_large_resolution_patch_embed_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_large_resolution_patch_embed");

    let input = b.add_input("image", &[IN_CHANNELS, LARGE_IMG_SIZE, LARGE_IMG_SIZE]);
    let weight = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let bias = b.add_input("patch_bias", &[HIDDEN_DIM]);

    let conv_out = b.add_conv2d(
        input,
        weight,
        Some(bias),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, LARGE_GRID_SIZE, LARGE_GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, LARGE_NUM_PATCHES]);
    let out = b.add_transpose(reshaped, &[1, 0], &[LARGE_NUM_PATCHES, HIDDEN_DIM]);

    b.build(out)
        .expect("valid large resolution patch embed kernel")
}

fn firered_large_resolution_patch_embed_bindings() -> Vec<TensorParamBinding> {
    let weight = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,               // image
        TensorParamBinding::ConstantTensor(weight), // patch_weight
        TensorParamBinding::ConstantTensor(bias),   // patch_bias
    ]
}

/// IBP through larger resolution patch embedding.
#[test]
fn test_firered_large_resolution_patch_embed_ibp() {
    let def = build_firered_large_resolution_patch_embed_kernel();
    let bindings = firered_large_resolution_patch_embed_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, LARGE_IMG_SIZE, LARGE_IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through large resolution patch embed");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[LARGE_NUM_PATCHES, HIDDEN_DIM],
        "large resolution patch embed output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR large resolution patch embed IBP ({LARGE_IMG_SIZE}x{LARGE_IMG_SIZE}, \
         {LARGE_NUM_PATCHES} patches): bounds=[{lo_min}, {hi_max}]"
    );
    assert!(lo_min.is_finite(), "large resolution lower must be finite");
    assert!(hi_max.is_finite(), "large resolution upper must be finite");
}

// ===========================================================================
// 49. Resolution scaling: large resolution -> encoder -> CTC (IBP)
// ===========================================================================

/// Build full pipeline at larger resolution: patch embed (42x42) -> encoder
/// -> CTC softmax.
fn build_firered_large_resolution_e2e_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_large_resolution_e2e");

    // Patch embedding at larger resolution
    let image = b.add_input("image", &[IN_CHANNELS, LARGE_IMG_SIZE, LARGE_IMG_SIZE]);
    let patch_w = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let patch_bias = b.add_input("patch_bias", &[HIDDEN_DIM]);
    let conv_out = b.add_conv2d(
        image,
        patch_w,
        Some(patch_bias),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, LARGE_GRID_SIZE, LARGE_GRID_SIZE],
    );
    let reshaped = b.add_reshape(conv_out, &[HIDDEN_DIM, LARGE_NUM_PATCHES]);
    let transposed = b.add_transpose(reshaped, &[1, 0], &[LARGE_NUM_PATCHES, HIDDEN_DIM]);

    // Single encoder layer
    let shape = [LARGE_NUM_PATCHES, HIDDEN_DIM];
    let ffn_shape = [LARGE_NUM_PATCHES, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(transposed, norm1_eps, 1, norm1_w, &shape);

    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(transposed, attn_out, &shape);

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
    let enc_out = b.add_binary_add(res1, ffn_out, &shape);

    // CTC head
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(
        enc_out,
        ctc_w,
        Some(ctc_bias),
        &[LARGE_NUM_PATCHES, VOCAB_SIZE],
    );
    let out = b.add_softmax(logits, -1, &[LARGE_NUM_PATCHES, VOCAB_SIZE]);

    b.build(out)
        .expect("valid large resolution end-to-end kernel")
}

fn firered_large_resolution_e2e_bindings() -> Vec<TensorParamBinding> {
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                       // image
        TensorParamBinding::ConstantTensor(patch_w),        // patch_weight
        TensorParamBinding::ConstantTensor(patch_bias),     // patch_bias
        TensorParamBinding::ConstantScalar(1e-5),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // q_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // k_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // v_weight
        TensorParamBinding::ConstantTensor(qkvo_w),         // out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
        TensorParamBinding::ConstantTensor(ctc_w),          // ctc_weight
        TensorParamBinding::ConstantTensor(ctc_bias),       // ctc_bias
    ]
}

/// IBP through large resolution end-to-end: image -> character probabilities.
#[test]
fn test_firered_large_resolution_e2e_ibp() {
    let def = build_firered_large_resolution_e2e_kernel();
    let bindings = firered_large_resolution_e2e_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, LARGE_IMG_SIZE, LARGE_IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through large resolution end-to-end");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[LARGE_NUM_PATCHES, VOCAB_SIZE],
        "large resolution e2e output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR large resolution e2e IBP ({LARGE_IMG_SIZE}x{LARGE_IMG_SIZE}): \
         bounds=[{lo_min}, {hi_max}]"
    );

    assert!(
        lo_min >= -1e-4,
        "large resolution e2e lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "large resolution e2e upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 50. Full OCR end-to-end CROWN: patch -> 8-layer encoder -> CTC
// ===========================================================================

/// CROWN through full OCR end-to-end pipeline.
///
/// Tests CROWN linearization through the deepest possible pipeline:
/// patch embedding -> 8-layer encoder -> CTC softmax. CROWN should produce
/// tighter bounds than IBP for the softmax output.
#[test]
fn test_firered_full_ocr_e2e_crown() {
    let def = build_firered_full_ocr_e2e_kernel();
    let bindings = firered_full_ocr_e2e_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, VOCAB_SIZE],
        "full OCR e2e CROWN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR full e2e CROWN (8 layers): method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Softmax terminal: output in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "full OCR e2e CROWN lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "full OCR e2e CROWN upper should be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 51. Four-layer encoder CROWN: CROWN at medium depth
// ===========================================================================

/// CROWN bounds through 4-layer encoder stack.
///
/// Bridges the gap between 2-layer (test 23) and 8-layer (test 35) CROWN
/// tests, verifying that CROWN linearization scales gracefully at medium depth.
#[test]
fn test_firered_four_layer_encoder_crown() {
    let def = build_firered_four_layer_encoder_kernel();
    let bindings = firered_four_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "FireRed-OCR four-layer encoder CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "four-layer CROWN lower must be finite");
    assert!(hi_max.is_finite(), "four-layer CROWN upper must be finite");
}

// ===========================================================================
// 52. Eight-layer encoder -> CTC pipeline IBP: deepest encoder + CTC
// ===========================================================================

/// Build 8-layer encoder -> CTC pipeline.
///
/// The deepest encoder-CTC composition: 8 encoder layers followed by
/// CTC projection + softmax. Verifies that probability bounds remain
/// in [0, 1] even at maximum depth.
fn build_firered_eight_layer_encoder_ctc_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("firered_ocr_eight_layer_encoder_ctc");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut prev = input;
    for i in 0..8 {
        let prefix = format!("b{i}");

        let norm1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let norm1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(prev, norm1_eps, 1, norm1_w, &shape);

        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, q_w, None, &shape);
        let k = b.add_linear(normed1, k_w, None, &shape);
        let v = b.add_linear(normed1, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(prev, attn_out, &shape);

        let norm2_eps = b.add_input(&format!("{prefix}_norm2_eps"), &[1]);
        let norm2_w = b.add_input(&format!("{prefix}_norm2_weight"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

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

    // CTC head
    let ctc_w = b.add_input("ctc_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let ctc_bias = b.add_input("ctc_bias", &[VOCAB_SIZE]);
    let logits = b.add_linear(prev, ctc_w, Some(ctc_bias), &[SEQ_LEN, VOCAB_SIZE]);
    let out = b.add_softmax(logits, -1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(out).expect("valid 8-layer encoder + CTC kernel")
}

fn firered_eight_layer_encoder_ctc_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let ctc_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);
    let ctc_bias = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE]), 0.0f32);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
    for _block in 0..8 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone()));
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone()));
    }
    bindings.push(TensorParamBinding::ConstantTensor(ctc_w));
    bindings.push(TensorParamBinding::ConstantTensor(ctc_bias));
    bindings
}

/// IBP through 8-layer encoder + CTC: character probabilities in [0, 1].
#[test]
fn test_firered_eight_layer_encoder_ctc_ibp() {
    let def = build_firered_eight_layer_encoder_ctc_kernel();
    let bindings = firered_eight_layer_encoder_ctc_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 8-layer encoder + CTC");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "8-layer encoder + CTC output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("FireRed-OCR 8-layer encoder + CTC IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(
        lo_min >= -1e-4,
        "8-layer encoder + CTC lower should be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "8-layer encoder + CTC upper should be <= 1, got {hi_max}"
    );
}
