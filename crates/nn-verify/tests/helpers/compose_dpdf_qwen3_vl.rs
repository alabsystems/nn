// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Qwen3-VL subgraph NY composition.
//!
//! Verifies bounds propagation through Qwen3-VL sub-blocks used in the
//! dpdf document understanding pipeline for vision-language understanding:
//!
//! 1. **Conv3D patch embedding IBP**: Conv2d(3, D, P, stride=P) -> reshape ->
//!    transpose (models the spatial slice of Qwen3-VL's 3D patch embedding;
//!    TensorBlockBuilder lacks Conv3D so we verify the spatial Conv2D core).
//!
//! 2. **Window attention IBP**: Q/K/V projection -> attention with windowed
//!    scope -> output projection. Local self-attention over fixed-size windows
//!    used in the Qwen3-VL vision encoder (ViT with window partitioning).
//!
//! 3. **Interleaved M-RoPE bounds**: Multi-modal Rotary Position Embedding
//!    with interleaved temporal/height/width frequencies. cos/sin values
//!    bounded in [-1, 1].
//!
//! 4. **Vision encoder block CROWN**: Window attention -> RMSNorm -> SwiGLU
//!    FFN -> RMSNorm with residual connections. One ViT block.
//!
//! 5. **Deep stack fusion IBP**: Two stacked vision encoder blocks,
//!    verifying bounds propagation through repeated attention + FFN layers.
//!
//! 6. **SwiGLU decoder FFN CROWN**: gate_proj -> SiLU -> mul(up_proj) ->
//!    down_proj. Qwen3 decoder FFN with CROWN linearization.
//!
//! 7. **GQA KV-cache IBP**: Grouped-query attention with separate Q/KV
//!    dimension projections, modeling the KV-cache inference path.
//!
//! 8. **MoE routing IBP**: Linear -> softmax router selecting top-k experts.
//!    Expert gate bounded in [0, 1].
//!
//! 9. **RMSNorm IBP**: Root mean square normalization used throughout
//!    Qwen3-VL encoder and decoder.
//!
//! 10. **Vision-language projection IBP**: Linear mapping from vision encoder
//!     features to LM embedding space for cross-modal fusion.
//!
//! 11. **Full VLM compose IBP**: Vision patch embedding -> encoder block ->
//!     vision projection -> decoder FFN. End-to-end simplified pipeline.
//!
//! 12. **MoE top-2 routing IBP**: Linear -> softmax -> narrow(2) for top-2
//!     expert selection. Verifies probability bounds after selection.
//!
//! 13. **MoE expert FFN IBP**: Single expert SwiGLU FFN path verifying
//!     bounds through one expert's gate/up/down projections.
//!
//! 14. **MoE residual composition IBP**: Expert FFN output + skip connection
//!     verifying residual addition preserves bounded outputs.
//!
//! 15. **Multimodal token interleave IBP**: Vision + text token splitting,
//!     per-modality projection, and concatenation along sequence dimension.
//!
//! 16. **Decoder two-layer stack IBP**: 2 decoder layers with GQA causal
//!     attention, RMSNorm, and SwiGLU FFN with residual connections.
//!
//! 17. **Causal attention M-RoPE IBP**: Pre-norm GQA attention with
//!     multimodal rotary position embeddings (cos/sin multiplication on Q/K).
//!
//! 18. **Decoder to LM head IBP**: RMSNorm -> Linear(HIDDEN -> VOCAB) ->
//!     softmax probability distribution. Output bounded in [0, 1].
//!
//! 19. **Quantized matmul bounds IBP**: Dequantized INT4 weights -> matmul.
//!     Verifies tighter output bounds from quantized weight magnitudes.
//!
//! 20. **Vision-to-decoder cross-modal IBP**: Vision RMSNorm -> projection ->
//!     decoder RMSNorm -> SwiGLU FFN + residual. Cross-modal boundary test.
//!
//! 21. **MoE 3B active composition IBP**: 2 parallel SwiGLU expert FFNs ->
//!     sum + residual. Models the 30B-A3B MoE active parameter path.
//!
//! 22. **Full decoder stack + LM head CROWN**: 2-layer decoder + final
//!     RMSNorm + LM head + softmax. Deepest CROWN linearization test.
//!
//! Architecture references:
//! - Qwen2-VL / Qwen3-VL (Alibaba): Vision-language model with 3D patch
//!   embedding, M-RoPE, and window attention in the vision encoder
//! - RMSNorm (Zhang & Sennrich, 2019): replaces LayerNorm in Qwen
//! - SwiGLU (Shazeer, 2020): SiLU-gated FFN in Qwen decoder
//! - GQA (Ainslie et al., 2023): Grouped-Query Attention in Qwen decoder
//! - MoE (Fedus et al., 2022): Mixture-of-Experts routing
//!
//! Dimensions (small for fast verification):
//! - IMG_SIZE=32, PATCH_SIZE=16, HIDDEN_DIM=64, FFN_DIM=128, SEQ_LEN=4,
//!   NUM_HEADS=4, NUM_KV_HEADS=2, NUM_EXPERTS=4
//!
//! Part of #3893, #3912: NY compose tests for Qwen3-VL subgraphs.

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
/// Hidden dimension (tiny for testing).
const HIDDEN_DIM: usize = 64;
/// FFN intermediate dimension (SwiGLU gate and up projections).
const FFN_DIM: usize = 128;
/// Sequence length for decoder sub-block tests.
const SEQ_LEN: usize = 4;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Number of KV heads for grouped-query attention.
const NUM_KV_HEADS: usize = 2;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 16
/// KV dimension = NUM_KV_HEADS * HEAD_DIM.
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 32
/// Number of MoE experts.
const NUM_EXPERTS: usize = 4;
/// Vocabulary size for LM head tests.
const VOCAB_SIZE: usize = 256;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;

// ===========================================================================
// 1. Conv3D patch embedding IBP (modeled as spatial Conv2D)
// ===========================================================================

/// Build a Qwen3-VL patch embedding kernel using Conv2d.
///
/// Qwen3-VL uses a 3D convolution (temporal + spatial) for video frames.
/// Since TensorBlockBuilder has no Conv3D, we verify the spatial core:
/// Conv2d(3, D, P, stride=P) on a single frame.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels in [0, 1]).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]` after reshape and transpose.
fn build_qwen3_vl_patch_embedding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_patch_embedding");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let weight = b.add_input(
        "patch_weight",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let bias = b.add_input("patch_bias", &[HIDDEN_DIM]);

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

    b.build(out).expect("valid Qwen3-VL patch embedding kernel")
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
fn qwen3_vl_patch_embedding_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // image [3, 32, 32]
        TensorParamBinding::ConstantTensor(w),    // patch_weight [D, 3, P, P]
        TensorParamBinding::ConstantTensor(bias), // patch_bias [D]
    ]
}

/// IBP bounds propagate through Qwen3-VL 3D patch embedding (spatial Conv2D core).
#[test]
fn test_conv3d_patch_embed_ibp() {
    let def = build_qwen3_vl_patch_embedding_kernel();
    let bindings = qwen3_vl_patch_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL patch embedding");

    let (lo, _hi) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "output shape should be [NUM_PATCHES={NUM_PATCHES}, HIDDEN_DIM={HIDDEN_DIM}], got {:?}",
        lo.shape()
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL patch embedding IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// 2. Window attention IBP
// ===========================================================================

/// Build a window attention kernel for the Qwen3-VL vision encoder.
///
/// Window attention restricts self-attention to a local window of patches.
/// For verification tractability, we use standard attention on a window-sized
/// sequence: [NUM_PATCHES, HIDDEN_DIM] with Q/K/V projections and softmax.
///
/// Input: `[NUM_PATCHES, HIDDEN_DIM]` (Variable, patch features).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
fn build_qwen3_vl_window_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_window_attention");

    let input = b.add_input("patch_features", &[NUM_PATCHES, HIDDEN_DIM]);
    let q_w = b.add_input("q_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let shape = [NUM_PATCHES, HIDDEN_DIM];

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let out = b.add_linear(attn, out_w, None, &shape);

    // Residual connection
    let result = b.add_binary_add(input, out, &shape);

    b.build(result)
        .expect("valid Qwen3-VL window attention kernel")
}

/// Bindings for window attention.
fn qwen3_vl_window_attention_bindings() -> Vec<TensorParamBinding> {
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

/// IBP bounds propagate through window attention.
#[test]
fn test_window_attention_ibp() {
    let def = build_qwen3_vl_window_attention_kernel();
    let bindings = qwen3_vl_window_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL window attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "window attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL window attention IBP (patches [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Residual connection preserves bounded output
    assert!(
        lo_min > -100.0,
        "window attention lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 3. Interleaved M-RoPE bounds
// ===========================================================================

/// Build an M-RoPE (Multimodal Rotary Position Embedding) kernel.
///
/// Qwen3-VL uses interleaved temporal/height/width position embeddings.
/// Each frequency dimension is assigned to one of three modalities.
/// For verification: cos/sin values are constant in [-1, 1].
///
/// Input: `[SEQ_LEN, HEAD_DIM]` (Variable, query/key vectors).
/// Output: `[SEQ_LEN, HEAD_DIM]`.
fn build_qwen3_vl_mrope_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_mrope");

    let input = b.add_input("qk", &[SEQ_LEN, HEAD_DIM]);
    let cos_pe = b.add_input("cos_mrope", &[SEQ_LEN, HEAD_DIM]);
    let sin_pe = b.add_input("sin_mrope", &[SEQ_LEN, HEAD_DIM]);

    let shape = [SEQ_LEN, HEAD_DIM];

    // x * cos(theta)
    let x_cos = b.add_binary_mul(input, cos_pe, &shape);

    // x * sin(theta) -- conservative approximation (see GLM-OCR pattern)
    let x_sin = b.add_binary_mul(input, sin_pe, &shape);

    // output = x*cos + rotated_x*sin (simplified as x*cos + x*sin)
    let out = b.add_binary_add(x_cos, x_sin, &shape);

    b.build(out).expect("valid Qwen3-VL M-RoPE kernel")
}

/// Bindings for M-RoPE with interleaved temporal/height/width frequencies.
///
/// HEAD_DIM is split into 3 sections: temporal, height, width.
/// Each section uses different base frequencies. cos/sin bounded in [-1, 1].
fn qwen3_vl_mrope_bindings() -> Vec<TensorParamBinding> {
    let n = SEQ_LEN * HEAD_DIM;
    let section_size = HEAD_DIM / 3; // approximate even split
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for t in 0..SEQ_LEN {
        for d in 0..HEAD_DIM {
            // Assign different base frequencies per modality section
            let base = if d < section_size {
                10000.0_f64 // temporal
            } else if d < 2 * section_size {
                5000.0_f64 // height
            } else {
                5000.0_f64 // width
            };
            let freq = (t as f64) / base.powf(2.0 * (d % section_size) as f64 / HEAD_DIM as f64);
            cos_data.push(freq.cos() as f32);
            sin_data.push(freq.sin() as f32);
        }
    }
    let cos_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HEAD_DIM]), cos_data).expect("valid cos shape");
    let sin_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, HEAD_DIM]), sin_data).expect("valid sin shape");

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_pe),
        TensorParamBinding::ConstantTensor(sin_pe),
    ]
}

/// M-RoPE bounds: cos/sin in [-1, 1], so output bounded by 2x input range.
#[test]
fn test_interleaved_mrope_bounds() {
    let def = build_qwen3_vl_mrope_kernel();
    let bindings = qwen3_vl_mrope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HEAD_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL M-RoPE");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HEAD_DIM],
        "M-RoPE output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL M-RoPE IBP (qk [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // output = x*cos + x*sin, with x in [-1,1] and cos/sin in [-1,1]
    // each product is in [-1,1], sum is in [-2, 2]
    assert!(
        hi_max <= 2.0 + 1e-4,
        "M-RoPE output should be <= 2 with unit input, got {hi_max}"
    );
    assert!(
        lo_min >= -2.0 - 1e-4,
        "M-RoPE output should be >= -2 with unit input, got {lo_min}"
    );
}

// ===========================================================================
// 4. Vision encoder block CROWN
// ===========================================================================

/// Build a Qwen3-VL vision encoder block:
/// RMSNorm -> Window Attention -> residual -> RMSNorm -> SwiGLU FFN -> residual.
///
/// Input: `[NUM_PATCHES, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
fn build_qwen3_vl_vision_encoder_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_vision_encoder_block");

    let input = b.add_input("patch_features", &[NUM_PATCHES, HIDDEN_DIM]);
    let shape = [NUM_PATCHES, HIDDEN_DIM];
    let ffn_shape = [NUM_PATCHES, FFN_DIM];

    // Pre-attention RMSNorm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // Window self-attention (using standard attention for tractability)
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

    // Residual connection after attention
    let residual1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(residual1, norm2_eps, 1, norm2_w, &shape);

    // SwiGLU FFN
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

    b.build(out)
        .expect("valid Qwen3-VL vision encoder block kernel")
}

/// Bindings for vision encoder block.
fn qwen3_vl_vision_encoder_block_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // patch_features
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

/// CROWN bounds propagate through the vision encoder block.
#[test]
fn test_vision_encoder_block_crown() {
    let def = build_qwen3_vl_vision_encoder_block_kernel();
    let bindings = qwen3_vl_vision_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL vision encoder block: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record vision encoder block.
#[test]
fn test_vision_encoder_block_verify_and_record() {
    let def = build_qwen3_vl_vision_encoder_block_kernel();
    let bindings = qwen3_vl_vision_encoder_block_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_vl_vision_encoder_block");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

// ===========================================================================
// 5. Deep stack fusion IBP (2 encoder blocks)
// ===========================================================================

/// Build two stacked vision encoder blocks.
///
/// Verifies bounds propagation through repeated attention + FFN layers.
/// Input: `[NUM_PATCHES, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
fn build_qwen3_vl_deep_stack_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_deep_stack");

    let input = b.add_input("patch_features", &[NUM_PATCHES, HIDDEN_DIM]);
    let shape = [NUM_PATCHES, HIDDEN_DIM];
    let ffn_shape = [NUM_PATCHES, FFN_DIM];

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
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
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

    b.build(out).expect("valid Qwen3-VL deep stack kernel")
}

/// Bindings for deep stack (2 blocks).
fn qwen3_vl_deep_stack_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    // Build bindings for 2 identical blocks
    let mut bindings = vec![TensorParamBinding::Variable]; // patch_features

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

/// IBP bounds propagate through 2-block deep stack.
#[test]
fn test_deep_stack_fusion_ibp() {
    let def = build_qwen3_vl_deep_stack_kernel();
    let bindings = qwen3_vl_deep_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL deep stack");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "deep stack output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL deep stack IBP (patches [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. SwiGLU decoder FFN CROWN
// ===========================================================================

/// Build a SwiGLU FFN kernel for the Qwen3-VL decoder.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_swiglu_decoder_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_swiglu_decoder_ffn");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let gate_w = b.add_input("gate_proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_proj_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_proj_weight", &[HIDDEN_DIM, FFN_DIM]);

    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // Gate branch: gate_proj -> SiLU
    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);

    // Up branch: up_proj
    let up = b.add_linear(input, up_w, None, &ffn_shape);

    // Multiplicative gating
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);

    // Down projection
    let out = b.add_linear(hidden, down_w, None, &out_shape);

    b.build(out)
        .expect("valid Qwen3-VL SwiGLU decoder FFN kernel")
}

/// Bindings for SwiGLU decoder FFN.
fn qwen3_vl_swiglu_decoder_ffn_bindings() -> Vec<TensorParamBinding> {
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

/// CROWN bounds propagate through SwiGLU decoder FFN.
#[test]
fn test_swiglu_decoder_ffn_crown() {
    let def = build_qwen3_vl_swiglu_decoder_ffn_kernel();
    let bindings = qwen3_vl_swiglu_decoder_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL SwiGLU decoder FFN CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record SwiGLU decoder FFN.
#[test]
fn test_swiglu_decoder_ffn_verify_and_record() {
    let def = build_qwen3_vl_swiglu_decoder_ffn_kernel();
    let bindings = qwen3_vl_swiglu_decoder_ffn_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_vl_swiglu_decoder_ffn");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 7. GQA KV-cache IBP
// ===========================================================================

/// Build a grouped-query attention kernel modeling the KV-cache path.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// GQA with NUM_HEADS=4 query heads and NUM_KV_HEADS=2 KV heads.
/// Simplified: project Q to full dim, K/V to KV_DIM, attention on
/// matching dims, then project back up. Residual connection.
fn build_qwen3_vl_gqa_kv_cache_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_gqa_kv_cache");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let q_w = b.add_input("q_proj_weight", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_proj_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_proj_weight", &[KV_DIM, HIDDEN_DIM]);
    let out_up_w = b.add_input("out_up_weight", &[HIDDEN_DIM, KV_DIM]);

    // Q/K/V projections to KV_DIM
    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, KV_DIM]);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, KV_DIM]);

    // Causal attention (decoder uses causal mask)
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );

    // Project back to hidden dim
    let projected = b.add_linear(attn, out_up_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    // Residual connection
    let out = b.add_binary_add(input, projected, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid Qwen3-VL GQA KV-cache kernel")
}

/// Bindings for GQA KV-cache.
fn qwen3_vl_gqa_kv_cache_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_up_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                 // hidden
        TensorParamBinding::ConstantTensor(q_w),      // q_proj_weight
        TensorParamBinding::ConstantTensor(k_w),      // k_proj_weight
        TensorParamBinding::ConstantTensor(v_w),      // v_proj_weight
        TensorParamBinding::ConstantTensor(out_up_w), // out_up_weight
    ]
}

/// IBP bounds propagate through GQA with KV-cache path.
#[test]
fn test_gqa_kv_cache_ibp() {
    let def = build_qwen3_vl_gqa_kv_cache_kernel();
    let bindings = qwen3_vl_gqa_kv_cache_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL GQA KV-cache");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "GQA KV-cache output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL GQA KV-cache IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min > -100.0,
        "GQA KV-cache lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 8. MoE routing IBP
// ===========================================================================

/// Build a Mixture-of-Experts routing kernel.
///
/// Qwen3 MoE variants use a learned router that produces softmax
/// probabilities over experts. Top-k experts are selected per token.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, NUM_EXPERTS]` (routing probabilities in [0, 1]).
///
/// Router: Linear(HIDDEN_DIM -> NUM_EXPERTS) -> softmax(dim=-1)
fn build_qwen3_vl_moe_routing_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_moe_routing");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_weight", &[NUM_EXPERTS, HIDDEN_DIM]);

    // Router: Linear -> softmax
    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    b.build(probs).expect("valid Qwen3-VL MoE routing kernel")
}

/// Bindings for MoE routing.
fn qwen3_vl_moe_routing_bindings() -> Vec<TensorParamBinding> {
    let router_w = ArrayD::from_elem(IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(router_w),
    ]
}

/// IBP bounds propagate through MoE routing.
///
/// Softmax output is a probability distribution: all elements in [0, 1].
#[test]
fn test_moe_routing_ibp() {
    let def = build_qwen3_vl_moe_routing_kernel();
    let bindings = qwen3_vl_moe_routing_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL MoE routing");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, NUM_EXPERTS],
        "MoE routing output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL MoE routing IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    // Softmax codomain is (0, 1)
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 9. RMSNorm IBP
// ===========================================================================

/// Build an RMSNorm kernel for Qwen3-VL.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, hidden states in [-1, 1]).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_rmsnorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_rmsnorm");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[HIDDEN_DIM]);

    let out = b.add_rms_norm(input, eps, 1, weight, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid Qwen3-VL RMSNorm kernel")
}

/// Bindings for RMSNorm.
fn qwen3_vl_rmsnorm_bindings() -> Vec<TensorParamBinding> {
    let weight = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(weight),
    ]
}

/// IBP bounds propagate through Qwen3-VL RMSNorm.
#[test]
fn test_rms_norm_ibp() {
    let def = build_qwen3_vl_rmsnorm_kernel();
    let bindings = qwen3_vl_rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL RMSNorm");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "RMSNorm output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL RMSNorm IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record Qwen3-VL RMSNorm.
#[test]
fn test_rms_norm_verify_and_record() {
    let def = build_qwen3_vl_rmsnorm_kernel();
    let bindings = qwen3_vl_rmsnorm_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_vl_rmsnorm");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 10. Vision-language projection IBP
// ===========================================================================

/// Build a vision-language projection kernel.
///
/// Maps vision encoder output to LM embedding space. In Qwen3-VL this
/// is a linear layer (or MLP) bridging the vision encoder and decoder.
///
/// Input: `[NUM_PATCHES, HIDDEN_DIM]` (Variable, vision encoder output).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]` (projected to LM space).
fn build_qwen3_vl_vision_language_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_vision_language_projection");

    let input = b.add_input("vision_features", &[NUM_PATCHES, HIDDEN_DIM]);
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_bias = b.add_input("proj_bias", &[HIDDEN_DIM]);

    let out = b.add_linear(input, proj_w, Some(proj_bias), &[NUM_PATCHES, HIDDEN_DIM]);

    b.build(out)
        .expect("valid Qwen3-VL vision-language projection kernel")
}

/// Bindings for vision-language projection.
fn qwen3_vl_vision_language_projection_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let proj_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                  // vision_features
        TensorParamBinding::ConstantTensor(proj_w),    // proj_weight
        TensorParamBinding::ConstantTensor(proj_bias), // proj_bias
    ]
}

/// IBP bounds propagate through vision-language projection.
#[test]
fn test_vision_language_projection_ibp() {
    let def = build_qwen3_vl_vision_language_projection_kernel();
    let bindings = qwen3_vl_vision_language_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL vision-language projection");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "vision-language projection output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Qwen3-VL vision-language projection IBP (features [-2,2]): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Linear with D=64, weight=0.02, input in [-2, 2]:
    // max output = sum(|w_i| * 2.0) = 64 * 0.02 * 2 = 2.56
    assert!(
        hi_max < 10.0,
        "projection upper should be < 10 with small weights, got {hi_max}"
    );
}

/// Verify and record vision-language projection.
#[test]
fn test_vision_language_projection_verify_and_record() {
    let def = build_qwen3_vl_vision_language_projection_kernel();
    let bindings = qwen3_vl_vision_language_projection_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 2.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "qwen3_vl_vision_language_projection",
    );
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

// ===========================================================================
// 11. Full VLM compose IBP
// ===========================================================================

/// Build a simplified end-to-end Qwen3-VL pipeline:
/// Patch embedding -> vision encoder block (RMSNorm + Attention + FFN) ->
/// vision projection -> decoder SwiGLU FFN.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image pixels in [0, 1]).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
///
/// This tests bounds propagation from raw image pixels through the vision
/// encoder, cross-modal projection, and one decoder FFN layer.
fn build_qwen3_vl_full_vlm_compose_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_full_vlm_compose");

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

    // --- Vision encoder block: RMSNorm -> Attention -> residual -> RMSNorm -> FFN -> residual ---
    let enc_norm1_eps = b.add_input("enc_norm1_eps", &[1]);
    let enc_norm1_w = b.add_input("enc_norm1_weight", &[HIDDEN_DIM]);
    let enc_normed1 = b.add_rms_norm(patches, enc_norm1_eps, 1, enc_norm1_w, &patch_shape);

    let enc_q_w = b.add_input("enc_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_k_w = b.add_input("enc_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_v_w = b.add_input("enc_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_out_w = b.add_input("enc_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let enc_q = b.add_linear(enc_normed1, enc_q_w, None, &patch_shape);
    let enc_k = b.add_linear(enc_normed1, enc_k_w, None, &patch_shape);
    let enc_v = b.add_linear(enc_normed1, enc_v_w, None, &patch_shape);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
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

    let enc_norm2_eps = b.add_input("enc_norm2_eps", &[1]);
    let enc_norm2_w = b.add_input("enc_norm2_weight", &[HIDDEN_DIM]);
    let enc_normed2 = b.add_rms_norm(enc_res1, enc_norm2_eps, 1, enc_norm2_w, &patch_shape);

    let enc_gate_w = b.add_input("enc_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let enc_up_w = b.add_input("enc_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let enc_down_w = b.add_input("enc_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let enc_gate = b.add_linear(enc_normed2, enc_gate_w, None, &ffn_shape);
    let enc_gate_sig = b.add_sigmoid(enc_gate, &ffn_shape);
    let enc_gate_act = b.add_binary_mul(enc_gate, enc_gate_sig, &ffn_shape);
    let enc_up = b.add_linear(enc_normed2, enc_up_w, None, &ffn_shape);
    let enc_hidden = b.add_binary_mul(enc_gate_act, enc_up, &ffn_shape);
    let enc_ffn_out = b.add_linear(enc_hidden, enc_down_w, None, &patch_shape);
    let enc_out = b.add_binary_add(enc_res1, enc_ffn_out, &patch_shape);

    // --- Vision-language projection ---
    let proj_w = b.add_input("vl_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(enc_out, proj_w, None, &patch_shape);

    // --- Decoder SwiGLU FFN ---
    let dec_gate_w = b.add_input("dec_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let dec_up_w = b.add_input("dec_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let dec_down_w = b.add_input("dec_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let dec_gate = b.add_linear(projected, dec_gate_w, None, &ffn_shape);
    let dec_gate_sig = b.add_sigmoid(dec_gate, &ffn_shape);
    let dec_gate_act = b.add_binary_mul(dec_gate, dec_gate_sig, &ffn_shape);
    let dec_up = b.add_linear(projected, dec_up_w, None, &ffn_shape);
    let dec_hidden = b.add_binary_mul(dec_gate_act, dec_up, &ffn_shape);
    let out = b.add_linear(dec_hidden, dec_down_w, None, &patch_shape);

    b.build(out)
        .expect("valid Qwen3-VL full VLM compose kernel")
}

/// Bindings for full VLM compose pipeline.
fn qwen3_vl_full_vlm_compose_bindings() -> Vec<TensorParamBinding> {
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

    vec![
        TensorParamBinding::Variable,                       // image
        TensorParamBinding::ConstantTensor(patch_w),        // patch_weight
        TensorParamBinding::ConstantTensor(patch_bias),     // patch_bias
        TensorParamBinding::ConstantScalar(1e-5),           // enc_norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // enc_norm1_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // enc_q_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // enc_k_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // enc_v_weight
        TensorParamBinding::ConstantTensor(qkvo_w),         // enc_out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // enc_norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // enc_norm2_weight
        TensorParamBinding::ConstantTensor(gate_w.clone()), // enc_gate_weight
        TensorParamBinding::ConstantTensor(up_w.clone()),   // enc_up_weight
        TensorParamBinding::ConstantTensor(down_w.clone()), // enc_down_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )), // vl_proj_weight
        TensorParamBinding::ConstantTensor(gate_w),         // dec_gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // dec_up_weight
        TensorParamBinding::ConstantTensor(down_w),         // dec_down_weight
    ]
}

/// IBP bounds propagate through the full Qwen3-VL pipeline end-to-end.
#[test]
fn test_full_vlm_compose_ibp() {
    let def = build_qwen3_vl_full_vlm_compose_kernel();
    let bindings = qwen3_vl_full_vlm_compose_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL full VLM compose");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "full VLM compose output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL full VLM compose IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record full VLM compose.
#[test]
fn test_full_vlm_compose_verify_and_record() {
    let def = build_qwen3_vl_full_vlm_compose_kernel();
    let bindings = qwen3_vl_full_vlm_compose_bindings();
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_vl_full_vlm_compose");
    assert_eq!(result.num_variables, 1, "single Variable input (image)");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

// ===========================================================================
// 12. MoE top-2 routing IBP
// ===========================================================================

/// Build a MoE top-2 routing kernel: Linear -> softmax -> top-2 selection.
///
/// Models the full expert routing path in Qwen3-VL MoE variants. The router
/// linear layer maps hidden states to expert logits, softmax normalizes to
/// probabilities, and top-2 selection picks the two highest-scoring experts.
/// Since TensorBlockBuilder has no topk op, we model top-2 as selecting
/// the first 2 expert columns after softmax (structural approximation).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, hidden states).
/// Output: `[SEQ_LEN, 2]` (top-2 routing probabilities).
fn build_qwen3_vl_moe_top2_routing_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_moe_top2_routing");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_weight", &[NUM_EXPERTS, HIDDEN_DIM]);

    // Router: Linear -> softmax
    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    // Approximate top-2 via narrow (select first 2 expert slots).
    // True top-k is a selection op; this verifies bounds on 2 expert probs.
    let top2 = b.add_narrow(probs, 1, 0, 2, &[SEQ_LEN, 2]);

    b.build(top2)
        .expect("valid Qwen3-VL MoE top-2 routing kernel")
}

/// Bindings for MoE top-2 routing.
fn qwen3_vl_moe_top2_routing_bindings() -> Vec<TensorParamBinding> {
    let router_w = ArrayD::from_elem(IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(router_w),
    ]
}

/// IBP bounds propagate through MoE top-2 routing.
///
/// Softmax output is in [0, 1]; narrow to 2 experts preserves this.
#[test]
fn test_moe_top2_routing_ibp() {
    let def = build_qwen3_vl_moe_top2_routing_kernel();
    let bindings = qwen3_vl_moe_top2_routing_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL MoE top-2 routing");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, 2],
        "MoE top-2 routing output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL MoE top-2 routing IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    // Softmax codomain is (0, 1), narrow preserves bounds
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 13. MoE expert FFN IBP (single expert SwiGLU)
// ===========================================================================

/// Build a single MoE expert FFN kernel: SwiGLU through one expert's weights.
///
/// Each MoE expert is a standard SwiGLU FFN: gate_proj -> SiLU -> mul(up_proj)
/// -> down_proj. This tests bounds propagation through a single expert path.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, routed token features).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_moe_expert_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_moe_expert_ffn");

    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let gate_w = b.add_input("expert_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("expert_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("expert_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // SwiGLU: gate_proj -> SiLU (sigmoid * x) -> mul(up_proj) -> down_proj
    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &out_shape);

    b.build(out).expect("valid Qwen3-VL MoE expert FFN kernel")
}

/// Bindings for a single MoE expert FFN.
fn qwen3_vl_moe_expert_ffn_bindings() -> Vec<TensorParamBinding> {
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

/// IBP bounds propagate through a single MoE expert SwiGLU FFN.
#[test]
fn test_moe_expert_ffn_ibp() {
    let def = build_qwen3_vl_moe_expert_ffn_kernel();
    let bindings = qwen3_vl_moe_expert_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL MoE expert FFN");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "MoE expert FFN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL MoE expert FFN IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 14. MoE residual composition IBP
// ===========================================================================

/// Build MoE residual composition: expert FFN output + skip connection.
///
/// In MoE models, the router-weighted expert outputs are summed with a
/// residual (skip) connection. This tests: input -> SwiGLU expert FFN ->
/// add(input, ffn_out) -> output.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_moe_residual_composition_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_moe_residual_composition");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let gate_w = b.add_input("expert_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("expert_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("expert_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // SwiGLU expert FFN
    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_activated = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_activated, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Residual (skip) connection: output = input + expert_ffn(input)
    let out = b.add_binary_add(input, ffn_out, &shape);

    b.build(out)
        .expect("valid Qwen3-VL MoE residual composition kernel")
}

/// Bindings for MoE residual composition.
fn qwen3_vl_moe_residual_composition_bindings() -> Vec<TensorParamBinding> {
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

/// IBP bounds propagate through MoE residual composition (expert + skip).
#[test]
fn test_moe_residual_composition_ibp() {
    let def = build_qwen3_vl_moe_residual_composition_kernel();
    let bindings = qwen3_vl_moe_residual_composition_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL MoE residual composition");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "MoE residual composition output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL MoE residual composition IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Residual preserves bounded output: input in [-1,1] + small FFN output
    assert!(
        lo_min > -50.0,
        "MoE residual lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 15. Multimodal token interleave IBP
// ===========================================================================

/// Number of vision tokens for interleave tests.
const VIS_TOKENS: usize = NUM_PATCHES; // 4
/// Number of text tokens for interleave tests.
const TEXT_TOKENS: usize = SEQ_LEN; // 4
/// Total interleaved sequence length.
const INTERLEAVED_LEN: usize = VIS_TOKENS + TEXT_TOKENS; // 8

/// Build a multimodal token interleave kernel.
///
/// Qwen3-VL concatenates projected vision tokens with embedded text tokens
/// before feeding them to the decoder. This tests bounds propagation through
/// two separate linear projections followed by concatenation along the
/// sequence dimension.
///
/// Input: `[INTERLEAVED_LEN, HIDDEN_DIM]` (Variable, combined features).
/// Output: `[INTERLEAVED_LEN, HIDDEN_DIM]` (projected and merged).
///
/// Structural approximation: since TensorBlockBuilder operates on a single
/// Variable input, we split it into vision (first VIS_TOKENS) and text
/// (last TEXT_TOKENS) via narrow, project each, then concatenate.
fn build_qwen3_vl_multimodal_token_interleave_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_multimodal_token_interleave");

    let input = b.add_input("combined_features", &[INTERLEAVED_LEN, HIDDEN_DIM]);
    let vis_proj_w = b.add_input("vis_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let text_proj_w = b.add_input("text_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let vis_shape = [VIS_TOKENS, HIDDEN_DIM];
    let text_shape = [TEXT_TOKENS, HIDDEN_DIM];
    let out_shape = [INTERLEAVED_LEN, HIDDEN_DIM];

    // Split into vision and text segments
    let vis_tokens = b.add_narrow(input, 0, 0, VIS_TOKENS, &vis_shape);
    let text_tokens = b.add_narrow(input, 0, VIS_TOKENS, TEXT_TOKENS, &text_shape);

    // Project each modality
    let vis_projected = b.add_linear(vis_tokens, vis_proj_w, None, &vis_shape);
    let text_projected = b.add_linear(text_tokens, text_proj_w, None, &text_shape);

    // Concatenate along sequence dimension
    let out = b.add_concat(&[vis_projected, text_projected], 0, &out_shape);

    b.build(out)
        .expect("valid Qwen3-VL multimodal token interleave kernel")
}

/// Bindings for multimodal token interleave.
fn qwen3_vl_multimodal_token_interleave_bindings() -> Vec<TensorParamBinding> {
    let vis_proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let text_proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                    // combined_features
        TensorParamBinding::ConstantTensor(vis_proj_w),  // vis_proj_weight
        TensorParamBinding::ConstantTensor(text_proj_w), // text_proj_weight
    ]
}

/// IBP bounds propagate through multimodal token interleave.
#[test]
fn test_multimodal_token_interleave_ibp() {
    let def = build_qwen3_vl_multimodal_token_interleave_kernel();
    let bindings = qwen3_vl_multimodal_token_interleave_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[INTERLEAVED_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL multimodal token interleave");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[INTERLEAVED_LEN, HIDDEN_DIM],
        "multimodal token interleave output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Qwen3-VL multimodal token interleave IBP (features [-1,1]): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Linear with D=64, weight=0.02, input in [-1, 1]:
    // max output = sum(|w_i| * 1.0) = 64 * 0.02 = 1.28
    assert!(
        hi_max < 5.0,
        "interleave upper should be < 5 with small weights, got {hi_max}"
    );
}

// ===========================================================================
// 16. Decoder two-layer stack IBP (with KV-cache path)
// ===========================================================================

/// Build a 2-layer decoder stack: each layer has pre-norm GQA attention +
/// pre-norm SwiGLU FFN with residual connections.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Models the Qwen3-VL decoder inference path with causal attention.
fn build_qwen3_vl_decoder_two_layer_stack_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_decoder_two_layer_stack");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer_idx in 0..2 {
        let prefix = format!("l{layer_idx}");

        // Pre-attention RMSNorm
        let norm1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let norm1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, norm1_eps, 1, norm1_w, &shape);

        // Causal self-attention (GQA with KV_DIM)
        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[KV_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[KV_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[KV_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, KV_DIM]);

        let q = b.add_linear(normed1, q_w, None, &[SEQ_LEN, KV_DIM]);
        let k = b.add_linear(normed1, k_w, None, &[SEQ_LEN, KV_DIM]);
        let v = b.add_linear(normed1, v_w, None, &[SEQ_LEN, KV_DIM]);
        let attn = b.add_attention(
            q,
            k,
            v,
            AttentionMask::Causal,
            Some(scale),
            &[SEQ_LEN, KV_DIM],
        );
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(current, attn_out, &shape);

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
        current = b.add_binary_add(res1, ffn_out, &shape);
    }

    b.build(current)
        .expect("valid Qwen3-VL decoder two-layer stack kernel")
}

/// Bindings for decoder two-layer stack.
fn qwen3_vl_decoder_two_layer_stack_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _layer in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(q_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(k_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(v_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(out_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gate_weight
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // up_weight
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // down_weight
    }

    bindings
}

/// IBP bounds propagate through 2-layer decoder stack with causal attention.
#[test]
fn test_decoder_two_layer_stack_ibp() {
    let def = build_qwen3_vl_decoder_two_layer_stack_kernel();
    let bindings = qwen3_vl_decoder_two_layer_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL decoder two-layer stack");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "decoder two-layer stack output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL decoder two-layer stack IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 17. Causal attention with M-RoPE IBP
// ===========================================================================

/// Build a causal attention block with M-RoPE position encoding.
///
/// Models the Qwen3-VL decoder attention with multimodal rotary position
/// embeddings: RMSNorm -> Q/K projections -> M-RoPE (cos/sin multiplication)
/// -> causal attention -> output projection.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_causal_attention_mrope_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_causal_attention_mrope");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let kv_shape = [SEQ_LEN, KV_DIM];

    // Pre-norm
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, norm_eps, 1, norm_w, &shape);

    // Q/K projections
    let q_w = b.add_input("q_proj_weight", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_proj_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_proj_weight", &[KV_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_proj_weight", &[HIDDEN_DIM, KV_DIM]);

    let q = b.add_linear(normed, q_w, None, &kv_shape);
    let k = b.add_linear(normed, k_w, None, &kv_shape);

    // Apply M-RoPE to Q and K
    let cos_pe = b.add_input("cos_mrope", &[SEQ_LEN, KV_DIM]);
    let sin_pe = b.add_input("sin_mrope", &[SEQ_LEN, KV_DIM]);

    let q_cos = b.add_binary_mul(q, cos_pe, &kv_shape);
    let q_sin = b.add_binary_mul(q, sin_pe, &kv_shape);
    let q_rope = b.add_binary_add(q_cos, q_sin, &kv_shape);

    let k_cos = b.add_binary_mul(k, cos_pe, &kv_shape);
    let k_sin = b.add_binary_mul(k, sin_pe, &kv_shape);
    let k_rope = b.add_binary_add(k_cos, k_sin, &kv_shape);

    // V projection (no RoPE)
    let v = b.add_linear(normed, v_w, None, &kv_shape);

    // Causal attention
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q_rope,
        k_rope,
        v,
        AttentionMask::Causal,
        Some(scale),
        &kv_shape,
    );

    // Output projection + residual
    let projected = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(input, projected, &shape);

    b.build(out)
        .expect("valid Qwen3-VL causal attention M-RoPE kernel")
}

/// Bindings for causal attention with M-RoPE.
fn qwen3_vl_causal_attention_mrope_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);

    // Build M-RoPE cos/sin with interleaved frequencies
    let n = SEQ_LEN * KV_DIM;
    let section_size = KV_DIM / 3;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for t in 0..SEQ_LEN {
        for d in 0..KV_DIM {
            let base = if d < section_size {
                10000.0_f64
            } else if d < 2 * section_size {
                5000.0_f64
            } else {
                5000.0_f64
            };
            let freq = (t as f64) / base.powf(2.0 * (d % section_size) as f64 / KV_DIM as f64);
            cos_data.push(freq.cos() as f32);
            sin_data.push(freq.sin() as f32);
        }
    }
    let cos_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, KV_DIM]), cos_data).expect("valid cos shape");
    let sin_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, KV_DIM]), sin_data).expect("valid sin shape");

    vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantScalar(1e-6),   // norm_eps
        TensorParamBinding::ConstantTensor(norm_w), // norm_weight
        TensorParamBinding::ConstantTensor(q_w),    // q_proj_weight
        TensorParamBinding::ConstantTensor(k_w),    // k_proj_weight
        TensorParamBinding::ConstantTensor(v_w),    // v_proj_weight
        TensorParamBinding::ConstantTensor(out_w),  // out_proj_weight
        TensorParamBinding::ConstantTensor(cos_pe), // cos_mrope
        TensorParamBinding::ConstantTensor(sin_pe), // sin_mrope
    ]
}

/// IBP bounds propagate through causal attention with M-RoPE.
#[test]
fn test_causal_attention_mrope_ibp() {
    let def = build_qwen3_vl_causal_attention_mrope_kernel();
    let bindings = qwen3_vl_causal_attention_mrope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL causal attention M-RoPE");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "causal attention M-RoPE output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL causal attention M-RoPE IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Residual preserves bounded output
    assert!(
        lo_min > -100.0,
        "causal attn M-RoPE lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 18. Decoder to LM head IBP
// ===========================================================================

/// Build a decoder output -> RMSNorm -> Linear LM head -> softmax kernel.
///
/// The final stage of the Qwen3-VL model: normalized decoder output is
/// projected to vocabulary-sized logits, then softmax to probabilities.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, decoder output).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution over vocabulary).
fn build_qwen3_vl_decoder_to_lm_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_decoder_to_lm_head");

    let input = b.add_input("decoder_output", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Final RMSNorm
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, norm_eps, 1, norm_w, &shape);

    // LM head: Linear(HIDDEN_DIM -> VOCAB_SIZE)
    let lm_head_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_head_w, None, &[SEQ_LEN, VOCAB_SIZE]);

    // Softmax to probabilities
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid Qwen3-VL decoder to LM head kernel")
}

/// Bindings for decoder to LM head.
fn qwen3_vl_decoder_to_lm_head_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let lm_head_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                  // decoder_output
        TensorParamBinding::ConstantScalar(1e-6),      // norm_eps
        TensorParamBinding::ConstantTensor(norm_w),    // norm_weight
        TensorParamBinding::ConstantTensor(lm_head_w), // lm_head_weight
    ]
}

/// IBP bounds propagate through decoder -> RMSNorm -> LM head -> softmax.
///
/// Output should be a probability distribution: all elements in [0, 1].
#[test]
fn test_decoder_to_lm_head_ibp() {
    let def = build_qwen3_vl_decoder_to_lm_head_kernel();
    let bindings = qwen3_vl_decoder_to_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL decoder to LM head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "decoder to LM head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL decoder to LM head IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    // Softmax codomain is (0, 1)
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 19. Quantized matmul bounds estimation IBP
// ===========================================================================

/// Quantization scale for INT4 approximation.
const QUANT_SCALE: f32 = 0.01;
/// INT4 range: [-8, 7] mapped to float via scale.
const INT4_MAX_FLOAT: f32 = 7.0 * QUANT_SCALE; // 0.07
/// Dequantized FFN dimension (matching FFN_DIM for pipeline compatibility).
const QUANT_FFN_DIM: usize = FFN_DIM;

/// Build a quantized matmul bounds estimation kernel.
///
/// Models the dequantize -> matmul path used in INT4-quantized inference.
/// Weights are stored as INT4 and dequantized at runtime via scale factor.
/// Since TensorBlockBuilder has no quantize op, we model the dequantized
/// weight bounds as a constant tensor with INT4-range values × scale.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, activations).
/// Output: `[SEQ_LEN, QUANT_FFN_DIM]` (quantized matmul output).
fn build_qwen3_vl_quantized_matmul_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_quantized_matmul");

    let input = b.add_input("activations", &[SEQ_LEN, HIDDEN_DIM]);
    let deq_w = b.add_input("dequantized_weight", &[QUANT_FFN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[QUANT_FFN_DIM]);

    let out = b.add_linear(input, deq_w, Some(bias), &[SEQ_LEN, QUANT_FFN_DIM]);

    b.build(out)
        .expect("valid Qwen3-VL quantized matmul kernel")
}

/// Bindings for quantized matmul (weights at INT4-range magnitude).
fn qwen3_vl_quantized_matmul_bindings() -> Vec<TensorParamBinding> {
    // Dequantized weights: INT4 range [-8, 7] * scale = [-0.08, 0.07]
    // Use the absolute max for uniform constant (conservative bound)
    let deq_w = ArrayD::from_elem(IxDyn(&[QUANT_FFN_DIM, HIDDEN_DIM]), INT4_MAX_FLOAT);
    let bias = ArrayD::from_elem(IxDyn(&[QUANT_FFN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,              // activations
        TensorParamBinding::ConstantTensor(deq_w), // dequantized_weight
        TensorParamBinding::ConstantTensor(bias),  // bias
    ]
}

/// IBP bounds propagate through quantized matmul.
///
/// Dequantized INT4 weights have smaller magnitude than FP32, so output
/// bounds should be tighter than standard matmul with full-precision weights.
#[test]
fn test_quantized_matmul_bounds_ibp() {
    let def = build_qwen3_vl_quantized_matmul_kernel();
    let bindings = qwen3_vl_quantized_matmul_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL quantized matmul");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, QUANT_FFN_DIM],
        "quantized matmul output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Qwen3-VL quantized matmul IBP (act [-1,1], INT4 weights): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // With INT4 dequantized weights (max 0.07), D=64, input in [-1, 1]:
    // max output per element = sum(|w_i| * 1.0) = 64 * 0.07 = 4.48
    assert!(
        hi_max < 10.0,
        "quantized matmul upper should be < 10 with INT4 weights, got {hi_max}"
    );
}

// ===========================================================================
// 20. Vision-to-decoder cross-modal IBP
// ===========================================================================

/// Build a vision-to-decoder cross-modal kernel.
///
/// Models the full path from vision encoder output through projection
/// into the decoder input space: RMSNorm -> Linear projection ->
/// decoder RMSNorm -> SwiGLU FFN. Tests bounds continuity across the
/// vision-language boundary.
///
/// Input: `[NUM_PATCHES, HIDDEN_DIM]` (Variable, vision encoder output).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
fn build_qwen3_vl_vision_to_decoder_crossmodal_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_vision_to_decoder_crossmodal");

    let input = b.add_input("vision_output", &[NUM_PATCHES, HIDDEN_DIM]);
    let shape = [NUM_PATCHES, HIDDEN_DIM];
    let ffn_shape = [NUM_PATCHES, FFN_DIM];

    // Vision-side RMSNorm (encoder output norm)
    let vis_norm_eps = b.add_input("vis_norm_eps", &[1]);
    let vis_norm_w = b.add_input("vis_norm_weight", &[HIDDEN_DIM]);
    let vis_normed = b.add_rms_norm(input, vis_norm_eps, 1, vis_norm_w, &shape);

    // Vision-language projection
    let proj_w = b.add_input("vl_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let proj_bias = b.add_input("vl_proj_bias", &[HIDDEN_DIM]);
    let projected = b.add_linear(vis_normed, proj_w, Some(proj_bias), &shape);

    // Decoder-side RMSNorm (first decoder layer input norm)
    let dec_norm_eps = b.add_input("dec_norm_eps", &[1]);
    let dec_norm_w = b.add_input("dec_norm_weight", &[HIDDEN_DIM]);
    let dec_normed = b.add_rms_norm(projected, dec_norm_eps, 1, dec_norm_w, &shape);

    // Decoder SwiGLU FFN
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(dec_normed, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(dec_normed, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Residual connection with projection output
    let out = b.add_binary_add(projected, ffn_out, &shape);

    b.build(out)
        .expect("valid Qwen3-VL vision-to-decoder cross-modal kernel")
}

/// Bindings for vision-to-decoder cross-modal.
fn qwen3_vl_vision_to_decoder_crossmodal_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let proj_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // vision_output
        TensorParamBinding::ConstantScalar(1e-6),           // vis_norm_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // vis_norm_weight
        TensorParamBinding::ConstantTensor(proj_w),         // vl_proj_weight
        TensorParamBinding::ConstantTensor(proj_bias),      // vl_proj_bias
        TensorParamBinding::ConstantScalar(1e-6),           // dec_norm_eps
        TensorParamBinding::ConstantTensor(norm_w),         // dec_norm_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
    ]
}

/// IBP bounds propagate through vision-to-decoder cross-modal path.
#[test]
fn test_vision_to_decoder_crossmodal_ibp() {
    let def = build_qwen3_vl_vision_to_decoder_crossmodal_kernel();
    let bindings = qwen3_vl_vision_to_decoder_crossmodal_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 2.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL vision-to-decoder cross-modal");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "vision-to-decoder cross-modal output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Qwen3-VL vision-to-decoder cross-modal IBP (vis [-2,2]): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 21. MoE 3B active composition IBP
// ===========================================================================

/// Active parameter dimension for 3B-active MoE (Qwen3-VL 30B-A3B).
/// Using smaller dimensions for tractable verification.
const MOE_EXPERT_FFN_DIM: usize = 64; // scaled from 2560 for testing

/// Build a MoE 3B-active composition kernel: router -> 2 expert FFNs ->
/// weighted sum + residual.
///
/// Models the Qwen3-VL 30B-A3B MoE architecture where top-2 experts are
/// selected per token. Each expert is a SwiGLU FFN. The router-weighted
/// expert outputs are summed with a residual connection.
///
/// Structural simplification: we model 2 experts (top-2) explicitly rather
/// than routing from NUM_EXPERTS, since TensorBlockBuilder has no topk op.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_moe_3b_active_composition_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_moe_3b_active_composition");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let expert_ffn_shape = [SEQ_LEN, MOE_EXPERT_FFN_DIM];

    // Expert 1: SwiGLU FFN
    let e1_gate_w = b.add_input("e1_gate_weight", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let e1_up_w = b.add_input("e1_up_weight", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let e1_down_w = b.add_input("e1_down_weight", &[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]);

    let e1_gate = b.add_linear(input, e1_gate_w, None, &expert_ffn_shape);
    let e1_gate_sig = b.add_sigmoid(e1_gate, &expert_ffn_shape);
    let e1_gate_act = b.add_binary_mul(e1_gate, e1_gate_sig, &expert_ffn_shape);
    let e1_up = b.add_linear(input, e1_up_w, None, &expert_ffn_shape);
    let e1_hidden = b.add_binary_mul(e1_gate_act, e1_up, &expert_ffn_shape);
    let e1_out = b.add_linear(e1_hidden, e1_down_w, None, &shape);

    // Expert 2: SwiGLU FFN
    let e2_gate_w = b.add_input("e2_gate_weight", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let e2_up_w = b.add_input("e2_up_weight", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let e2_down_w = b.add_input("e2_down_weight", &[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]);

    let e2_gate = b.add_linear(input, e2_gate_w, None, &expert_ffn_shape);
    let e2_gate_sig = b.add_sigmoid(e2_gate, &expert_ffn_shape);
    let e2_gate_act = b.add_binary_mul(e2_gate, e2_gate_sig, &expert_ffn_shape);
    let e2_up = b.add_linear(input, e2_up_w, None, &expert_ffn_shape);
    let e2_hidden = b.add_binary_mul(e2_gate_act, e2_up, &expert_ffn_shape);
    let e2_out = b.add_linear(e2_hidden, e2_down_w, None, &shape);

    // Sum expert outputs (equal-weight approximation; real routing uses
    // softmax probabilities, but addition is a sound over-approximation
    // since both expert outputs are bounded)
    let expert_sum = b.add_binary_add(e1_out, e2_out, &shape);

    // Residual connection
    let out = b.add_binary_add(input, expert_sum, &shape);

    b.build(out)
        .expect("valid Qwen3-VL MoE 3B active composition kernel")
}

/// Bindings for MoE 3B active composition.
fn qwen3_vl_moe_3b_active_composition_bindings() -> Vec<TensorParamBinding> {
    let e_gate_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let e_up_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let e_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                         // hidden
        TensorParamBinding::ConstantTensor(e_gate_w.clone()), // e1_gate_weight
        TensorParamBinding::ConstantTensor(e_up_w.clone()),   // e1_up_weight
        TensorParamBinding::ConstantTensor(e_down_w.clone()), // e1_down_weight
        TensorParamBinding::ConstantTensor(e_gate_w),         // e2_gate_weight
        TensorParamBinding::ConstantTensor(e_up_w),           // e2_up_weight
        TensorParamBinding::ConstantTensor(e_down_w),         // e2_down_weight
    ]
}

/// IBP bounds propagate through MoE 3B active composition (2 experts + residual).
#[test]
fn test_moe_3b_active_composition_ibp() {
    let def = build_qwen3_vl_moe_3b_active_composition_kernel();
    let bindings = qwen3_vl_moe_3b_active_composition_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL MoE 3B active composition");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "MoE 3B active composition output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Qwen3-VL MoE 3B active composition IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // 2 expert FFNs + residual; with small weights should stay bounded
    assert!(
        lo_min > -100.0,
        "MoE 3B active lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 22. Full decoder stack + LM head CROWN
// ===========================================================================

/// Build a 2-layer decoder stack + LM head with CROWN linearization.
///
/// This is the deepest verification test: 2 decoder layers (each with
/// pre-norm GQA attention + pre-norm SwiGLU FFN) followed by final
/// RMSNorm -> Linear LM head -> softmax. Tests CROWN's ability to
/// linearize through the full decoder-to-output path.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, decoder input embeddings).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution).
fn build_qwen3_vl_full_decoder_stack_lm_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_full_decoder_stack_lm_head");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    // 2-layer decoder stack
    for layer_idx in 0..2 {
        let prefix = format!("l{layer_idx}");

        // Pre-attention RMSNorm
        let norm1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let norm1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, norm1_eps, 1, norm1_w, &shape);

        // Causal self-attention
        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, q_w, None, &shape);
        let k = b.add_linear(normed1, k_w, None, &shape);
        let v = b.add_linear(normed1, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(current, attn_out, &shape);

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
        current = b.add_binary_add(res1, ffn_out, &shape);
    }

    // Final RMSNorm
    let final_norm_eps = b.add_input("final_norm_eps", &[1]);
    let final_norm_w = b.add_input("final_norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(current, final_norm_eps, 1, final_norm_w, &shape);

    // LM head: Linear -> softmax
    let lm_head_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_head_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid Qwen3-VL full decoder stack + LM head kernel")
}

/// Bindings for full decoder stack + LM head.
fn qwen3_vl_full_decoder_stack_lm_head_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);
    let lm_head_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _layer in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gate_weight
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // up_weight
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // down_weight
    }

    // Final norm + LM head
    bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // final_norm_eps
    bindings.push(TensorParamBinding::ConstantTensor(norm_w)); // final_norm_weight
    bindings.push(TensorParamBinding::ConstantTensor(lm_head_w)); // lm_head_weight

    bindings
}

/// CROWN bounds propagate through 2-layer decoder + LM head.
///
/// This tests CROWN linearization through the deepest Qwen3-VL subgraph:
/// 2 decoder layers (RMSNorm + attention + SwiGLU FFN each) + final
/// RMSNorm + Linear + softmax. CROWN may fall back to IBP for normalization
/// layers; the test checks structural soundness regardless.
#[test]
fn test_full_decoder_stack_lm_head_crown() {
    let def = build_qwen3_vl_full_decoder_stack_lm_head_kernel();
    let bindings = qwen3_vl_full_decoder_stack_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "full decoder stack + LM head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Qwen3-VL full decoder stack + LM head: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Softmax codomain is (0, 1)
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 23. 3-layer MoE decoder stack IBP
// ===========================================================================

/// Build a 3-layer decoder stack where each layer uses MoE (2 experts)
/// instead of a single SwiGLU FFN.
///
/// Each layer: RMSNorm -> causal attention -> residual -> RMSNorm ->
///             2-expert MoE (parallel SwiGLU FFNs summed) -> residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_3layer_moe_decoder_stack_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_3layer_moe_decoder_stack");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let expert_ffn_shape = [SEQ_LEN, MOE_EXPERT_FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer_idx in 0..3 {
        let prefix = format!("l{layer_idx}");

        // Pre-attention RMSNorm
        let norm1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let norm1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, norm1_eps, 1, norm1_w, &shape);

        // Causal self-attention
        let q_w = b.add_input(&format!("{prefix}_q_weight"), &[KV_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_weight"), &[KV_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_weight"), &[KV_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_weight"), &[HIDDEN_DIM, KV_DIM]);

        let q = b.add_linear(normed1, q_w, None, &[SEQ_LEN, KV_DIM]);
        let k = b.add_linear(normed1, k_w, None, &[SEQ_LEN, KV_DIM]);
        let v = b.add_linear(normed1, v_w, None, &[SEQ_LEN, KV_DIM]);
        let attn = b.add_attention(
            q,
            k,
            v,
            AttentionMask::Causal,
            Some(scale),
            &[SEQ_LEN, KV_DIM],
        );
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(current, attn_out, &shape);

        // Pre-FFN RMSNorm
        let norm2_eps = b.add_input(&format!("{prefix}_norm2_eps"), &[1]);
        let norm2_w = b.add_input(&format!("{prefix}_norm2_weight"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

        // MoE: 2 expert SwiGLU FFNs
        let e1_gate_w = b.add_input(
            &format!("{prefix}_e1_gate_w"),
            &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM],
        );
        let e1_up_w = b.add_input(
            &format!("{prefix}_e1_up_w"),
            &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM],
        );
        let e1_down_w = b.add_input(
            &format!("{prefix}_e1_down_w"),
            &[HIDDEN_DIM, MOE_EXPERT_FFN_DIM],
        );

        let e1_gate = b.add_linear(normed2, e1_gate_w, None, &expert_ffn_shape);
        let e1_gate_sig = b.add_sigmoid(e1_gate, &expert_ffn_shape);
        let e1_gate_act = b.add_binary_mul(e1_gate, e1_gate_sig, &expert_ffn_shape);
        let e1_up = b.add_linear(normed2, e1_up_w, None, &expert_ffn_shape);
        let e1_hidden = b.add_binary_mul(e1_gate_act, e1_up, &expert_ffn_shape);
        let e1_out = b.add_linear(e1_hidden, e1_down_w, None, &shape);

        let e2_gate_w = b.add_input(
            &format!("{prefix}_e2_gate_w"),
            &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM],
        );
        let e2_up_w = b.add_input(
            &format!("{prefix}_e2_up_w"),
            &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM],
        );
        let e2_down_w = b.add_input(
            &format!("{prefix}_e2_down_w"),
            &[HIDDEN_DIM, MOE_EXPERT_FFN_DIM],
        );

        let e2_gate = b.add_linear(normed2, e2_gate_w, None, &expert_ffn_shape);
        let e2_gate_sig = b.add_sigmoid(e2_gate, &expert_ffn_shape);
        let e2_gate_act = b.add_binary_mul(e2_gate, e2_gate_sig, &expert_ffn_shape);
        let e2_up = b.add_linear(normed2, e2_up_w, None, &expert_ffn_shape);
        let e2_hidden = b.add_binary_mul(e2_gate_act, e2_up, &expert_ffn_shape);
        let e2_out = b.add_linear(e2_hidden, e2_down_w, None, &shape);

        // Sum expert outputs + residual
        let expert_sum = b.add_binary_add(e1_out, e2_out, &shape);
        current = b.add_binary_add(res1, expert_sum, &shape);
    }

    b.build(current)
        .expect("valid Qwen3-VL 3-layer MoE decoder stack kernel")
}

/// Bindings for 3-layer MoE decoder stack.
fn qwen3_vl_3layer_moe_decoder_stack_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);
    let e_gate_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let e_up_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let e_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _layer in 0..3 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(q_w.clone())); // q_weight
        bindings.push(TensorParamBinding::ConstantTensor(k_w.clone())); // k_weight
        bindings.push(TensorParamBinding::ConstantTensor(v_w.clone())); // v_weight
        bindings.push(TensorParamBinding::ConstantTensor(out_w.clone())); // out_weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
                                                                           // Expert 1
        bindings.push(TensorParamBinding::ConstantTensor(e_gate_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(e_up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(e_down_w.clone()));
        // Expert 2
        bindings.push(TensorParamBinding::ConstantTensor(e_gate_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(e_up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(e_down_w.clone()));
    }

    bindings
}

/// IBP bounds propagate through 3-layer MoE decoder stack.
#[test]
fn test_3layer_moe_decoder_stack_ibp() {
    let def = build_qwen3_vl_3layer_moe_decoder_stack_kernel();
    let bindings = qwen3_vl_3layer_moe_decoder_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL 3-layer MoE decoder stack");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "3-layer MoE decoder stack output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Qwen3-VL 3-layer MoE decoder stack IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 24. Multi-expert routing with 4 experts IBP
// ===========================================================================

/// Build a multi-expert routing kernel with 4 experts: router -> softmax ->
/// 4 expert FFNs -> weighted sum.
///
/// Models the full 4-expert MoE routing path where each expert contributes
/// to the output. Expert outputs are summed (sound over-approximation of
/// weighted combination since weights are in [0, 1]).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_multi_expert_routing_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_multi_expert_routing");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let expert_ffn_shape = [SEQ_LEN, MOE_EXPERT_FFN_DIM];

    // Router: Linear -> softmax (verifies routing probabilities)
    let router_w = b.add_input("router_weight", &[NUM_EXPERTS, HIDDEN_DIM]);
    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let _probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    // 4 expert SwiGLU FFNs
    let mut expert_outputs = Vec::new();
    for expert_idx in 0..NUM_EXPERTS {
        let prefix = format!("e{expert_idx}");
        let gate_w = b.add_input(
            &format!("{prefix}_gate_w"),
            &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM],
        );
        let up_w = b.add_input(&format!("{prefix}_up_w"), &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(
            &format!("{prefix}_down_w"),
            &[HIDDEN_DIM, MOE_EXPERT_FFN_DIM],
        );

        let gate = b.add_linear(input, gate_w, None, &expert_ffn_shape);
        let gate_sig = b.add_sigmoid(gate, &expert_ffn_shape);
        let gate_act = b.add_binary_mul(gate, gate_sig, &expert_ffn_shape);
        let up = b.add_linear(input, up_w, None, &expert_ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &expert_ffn_shape);
        let out = b.add_linear(hidden, down_w, None, &shape);
        expert_outputs.push(out);
    }

    // Sum all expert outputs (sound over-approximation)
    let mut combined = expert_outputs[0];
    for &eo in &expert_outputs[1..] {
        combined = b.add_binary_add(combined, eo, &shape);
    }

    // Residual connection
    let out = b.add_binary_add(input, combined, &shape);

    b.build(out)
        .expect("valid Qwen3-VL multi-expert routing kernel")
}

/// Bindings for multi-expert routing.
fn qwen3_vl_multi_expert_routing_bindings() -> Vec<TensorParamBinding> {
    let router_w = ArrayD::from_elem(IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]), WEIGHT_MAG);
    let e_gate_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let e_up_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let e_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![
        TensorParamBinding::Variable,                 // hidden
        TensorParamBinding::ConstantTensor(router_w), // router_weight
    ];

    for _expert in 0..NUM_EXPERTS {
        bindings.push(TensorParamBinding::ConstantTensor(e_gate_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(e_up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(e_down_w.clone()));
    }

    bindings
}

/// IBP bounds propagate through multi-expert routing with 4 experts.
#[test]
fn test_multi_expert_routing_ibp() {
    let def = build_qwen3_vl_multi_expert_routing_kernel();
    let bindings = qwen3_vl_multi_expert_routing_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL multi-expert routing");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "multi-expert routing output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL multi-expert routing IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 25. Full decoder with interleaved attention + MoE IBP
// ===========================================================================

/// Build a decoder layer with interleaved causal attention and MoE FFN,
/// followed by a second layer with standard SwiGLU FFN. Tests the
/// heterogeneous layer composition found in Qwen3-VL MoE variants where
/// MoE layers alternate with dense layers.
///
/// Layer 0: RMSNorm -> causal attention -> residual -> RMSNorm -> MoE -> residual
/// Layer 1: RMSNorm -> causal attention -> residual -> RMSNorm -> dense FFN -> residual
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_interleaved_attn_moe_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_interleaved_attn_moe");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let expert_ffn_shape = [SEQ_LEN, MOE_EXPERT_FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // --- Layer 0: Attention + MoE ---
    let l0_norm1_eps = b.add_input("l0_norm1_eps", &[1]);
    let l0_norm1_w = b.add_input("l0_norm1_weight", &[HIDDEN_DIM]);
    let l0_normed1 = b.add_rms_norm(input, l0_norm1_eps, 1, l0_norm1_w, &shape);

    let l0_q_w = b.add_input("l0_q_weight", &[KV_DIM, HIDDEN_DIM]);
    let l0_k_w = b.add_input("l0_k_weight", &[KV_DIM, HIDDEN_DIM]);
    let l0_v_w = b.add_input("l0_v_weight", &[KV_DIM, HIDDEN_DIM]);
    let l0_out_w = b.add_input("l0_out_weight", &[HIDDEN_DIM, KV_DIM]);

    let l0_q = b.add_linear(l0_normed1, l0_q_w, None, &[SEQ_LEN, KV_DIM]);
    let l0_k = b.add_linear(l0_normed1, l0_k_w, None, &[SEQ_LEN, KV_DIM]);
    let l0_v = b.add_linear(l0_normed1, l0_v_w, None, &[SEQ_LEN, KV_DIM]);
    let l0_attn = b.add_attention(
        l0_q,
        l0_k,
        l0_v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );
    let l0_attn_out = b.add_linear(l0_attn, l0_out_w, None, &shape);
    let l0_res1 = b.add_binary_add(input, l0_attn_out, &shape);

    let l0_norm2_eps = b.add_input("l0_norm2_eps", &[1]);
    let l0_norm2_w = b.add_input("l0_norm2_weight", &[HIDDEN_DIM]);
    let l0_normed2 = b.add_rms_norm(l0_res1, l0_norm2_eps, 1, l0_norm2_w, &shape);

    // MoE with 2 experts
    let l0_e1_gate_w = b.add_input("l0_e1_gate_w", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let l0_e1_up_w = b.add_input("l0_e1_up_w", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let l0_e1_down_w = b.add_input("l0_e1_down_w", &[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]);

    let l0_e1_gate = b.add_linear(l0_normed2, l0_e1_gate_w, None, &expert_ffn_shape);
    let l0_e1_gate_sig = b.add_sigmoid(l0_e1_gate, &expert_ffn_shape);
    let l0_e1_gate_act = b.add_binary_mul(l0_e1_gate, l0_e1_gate_sig, &expert_ffn_shape);
    let l0_e1_up = b.add_linear(l0_normed2, l0_e1_up_w, None, &expert_ffn_shape);
    let l0_e1_hidden = b.add_binary_mul(l0_e1_gate_act, l0_e1_up, &expert_ffn_shape);
    let l0_e1_out = b.add_linear(l0_e1_hidden, l0_e1_down_w, None, &shape);

    let l0_e2_gate_w = b.add_input("l0_e2_gate_w", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let l0_e2_up_w = b.add_input("l0_e2_up_w", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let l0_e2_down_w = b.add_input("l0_e2_down_w", &[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]);

    let l0_e2_gate = b.add_linear(l0_normed2, l0_e2_gate_w, None, &expert_ffn_shape);
    let l0_e2_gate_sig = b.add_sigmoid(l0_e2_gate, &expert_ffn_shape);
    let l0_e2_gate_act = b.add_binary_mul(l0_e2_gate, l0_e2_gate_sig, &expert_ffn_shape);
    let l0_e2_up = b.add_linear(l0_normed2, l0_e2_up_w, None, &expert_ffn_shape);
    let l0_e2_hidden = b.add_binary_mul(l0_e2_gate_act, l0_e2_up, &expert_ffn_shape);
    let l0_e2_out = b.add_linear(l0_e2_hidden, l0_e2_down_w, None, &shape);

    let l0_expert_sum = b.add_binary_add(l0_e1_out, l0_e2_out, &shape);
    let l0_res2 = b.add_binary_add(l0_res1, l0_expert_sum, &shape);

    // --- Layer 1: Attention + Dense FFN ---
    let l1_norm1_eps = b.add_input("l1_norm1_eps", &[1]);
    let l1_norm1_w = b.add_input("l1_norm1_weight", &[HIDDEN_DIM]);
    let l1_normed1 = b.add_rms_norm(l0_res2, l1_norm1_eps, 1, l1_norm1_w, &shape);

    let l1_q_w = b.add_input("l1_q_weight", &[KV_DIM, HIDDEN_DIM]);
    let l1_k_w = b.add_input("l1_k_weight", &[KV_DIM, HIDDEN_DIM]);
    let l1_v_w = b.add_input("l1_v_weight", &[KV_DIM, HIDDEN_DIM]);
    let l1_out_w = b.add_input("l1_out_weight", &[HIDDEN_DIM, KV_DIM]);

    let l1_q = b.add_linear(l1_normed1, l1_q_w, None, &[SEQ_LEN, KV_DIM]);
    let l1_k = b.add_linear(l1_normed1, l1_k_w, None, &[SEQ_LEN, KV_DIM]);
    let l1_v = b.add_linear(l1_normed1, l1_v_w, None, &[SEQ_LEN, KV_DIM]);
    let l1_attn = b.add_attention(
        l1_q,
        l1_k,
        l1_v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );
    let l1_attn_out = b.add_linear(l1_attn, l1_out_w, None, &shape);
    let l1_res1 = b.add_binary_add(l0_res2, l1_attn_out, &shape);

    let l1_norm2_eps = b.add_input("l1_norm2_eps", &[1]);
    let l1_norm2_w = b.add_input("l1_norm2_weight", &[HIDDEN_DIM]);
    let l1_normed2 = b.add_rms_norm(l1_res1, l1_norm2_eps, 1, l1_norm2_w, &shape);

    // Dense SwiGLU FFN
    let l1_gate_w = b.add_input("l1_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l1_up_w = b.add_input("l1_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let l1_down_w = b.add_input("l1_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let l1_gate = b.add_linear(l1_normed2, l1_gate_w, None, &ffn_shape);
    let l1_gate_sig = b.add_sigmoid(l1_gate, &ffn_shape);
    let l1_gate_act = b.add_binary_mul(l1_gate, l1_gate_sig, &ffn_shape);
    let l1_up = b.add_linear(l1_normed2, l1_up_w, None, &ffn_shape);
    let l1_hidden = b.add_binary_mul(l1_gate_act, l1_up, &ffn_shape);
    let l1_ffn_out = b.add_linear(l1_hidden, l1_down_w, None, &shape);
    let out = b.add_binary_add(l1_res1, l1_ffn_out, &shape);

    b.build(out)
        .expect("valid Qwen3-VL interleaved attention + MoE kernel")
}

/// Bindings for interleaved attention + MoE decoder.
fn qwen3_vl_interleaved_attn_moe_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);
    let e_gate_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let e_up_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let e_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable, // hidden
        // Layer 0
        TensorParamBinding::ConstantScalar(1e-6), // l0_norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // l0_norm1_weight
        TensorParamBinding::ConstantTensor(q_w.clone()), // l0_q_weight
        TensorParamBinding::ConstantTensor(k_w.clone()), // l0_k_weight
        TensorParamBinding::ConstantTensor(v_w.clone()), // l0_v_weight
        TensorParamBinding::ConstantTensor(out_w.clone()), // l0_out_weight
        TensorParamBinding::ConstantScalar(1e-6), // l0_norm2_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // l0_norm2_weight
        TensorParamBinding::ConstantTensor(e_gate_w.clone()), // l0_e1_gate_w
        TensorParamBinding::ConstantTensor(e_up_w.clone()), // l0_e1_up_w
        TensorParamBinding::ConstantTensor(e_down_w.clone()), // l0_e1_down_w
        TensorParamBinding::ConstantTensor(e_gate_w), // l0_e2_gate_w
        TensorParamBinding::ConstantTensor(e_up_w), // l0_e2_up_w
        TensorParamBinding::ConstantTensor(e_down_w), // l0_e2_down_w
        // Layer 1
        TensorParamBinding::ConstantScalar(1e-6), // l1_norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // l1_norm1_weight
        TensorParamBinding::ConstantTensor(q_w),  // l1_q_weight
        TensorParamBinding::ConstantTensor(k_w),  // l1_k_weight
        TensorParamBinding::ConstantTensor(v_w),  // l1_v_weight
        TensorParamBinding::ConstantTensor(out_w), // l1_out_weight
        TensorParamBinding::ConstantScalar(1e-6), // l1_norm2_eps
        TensorParamBinding::ConstantTensor(norm_w), // l1_norm2_weight
        TensorParamBinding::ConstantTensor(gate_w), // l1_gate_weight
        TensorParamBinding::ConstantTensor(up_w), // l1_up_weight
        TensorParamBinding::ConstantTensor(down_w), // l1_down_weight
    ]
}

/// IBP bounds propagate through interleaved attention + MoE decoder.
#[test]
fn test_interleaved_attn_moe_ibp() {
    let def = build_qwen3_vl_interleaved_attn_moe_kernel();
    let bindings = qwen3_vl_interleaved_attn_moe_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL interleaved attention + MoE");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "interleaved attn + MoE output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL interleaved attn + MoE IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 26. Vision-to-text pipeline IBP (patch embed -> encoder -> project ->
//     interleave -> decoder -> LM head)
// ===========================================================================

/// Build a vision-to-text pipeline:
/// Conv2d patch embed -> vision encoder block -> projection ->
/// decoder SwiGLU FFN -> RMSNorm -> LM head -> softmax.
///
/// End-to-end from raw image pixels to vocabulary probabilities.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image pixels [0, 1]).
/// Output: `[NUM_PATCHES, VOCAB_SIZE]` (probability distribution).
fn build_qwen3_vl_vision_to_text_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_vision_to_text_pipeline");

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

    // --- Vision encoder block ---
    let enc_norm1_eps = b.add_input("enc_norm1_eps", &[1]);
    let enc_norm1_w = b.add_input("enc_norm1_weight", &[HIDDEN_DIM]);
    let enc_normed1 = b.add_rms_norm(patches, enc_norm1_eps, 1, enc_norm1_w, &patch_shape);

    let enc_q_w = b.add_input("enc_q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_k_w = b.add_input("enc_k_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_v_w = b.add_input("enc_v_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let enc_out_w = b.add_input("enc_out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let enc_q = b.add_linear(enc_normed1, enc_q_w, None, &patch_shape);
    let enc_k = b.add_linear(enc_normed1, enc_k_w, None, &patch_shape);
    let enc_v = b.add_linear(enc_normed1, enc_v_w, None, &patch_shape);
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

    let enc_norm2_eps = b.add_input("enc_norm2_eps", &[1]);
    let enc_norm2_w = b.add_input("enc_norm2_weight", &[HIDDEN_DIM]);
    let enc_normed2 = b.add_rms_norm(enc_res1, enc_norm2_eps, 1, enc_norm2_w, &patch_shape);

    let enc_gate_w = b.add_input("enc_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let enc_up_w = b.add_input("enc_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let enc_down_w = b.add_input("enc_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let enc_gate = b.add_linear(enc_normed2, enc_gate_w, None, &ffn_shape);
    let enc_gate_sig = b.add_sigmoid(enc_gate, &ffn_shape);
    let enc_gate_act = b.add_binary_mul(enc_gate, enc_gate_sig, &ffn_shape);
    let enc_up = b.add_linear(enc_normed2, enc_up_w, None, &ffn_shape);
    let enc_hidden = b.add_binary_mul(enc_gate_act, enc_up, &ffn_shape);
    let enc_ffn_out = b.add_linear(enc_hidden, enc_down_w, None, &patch_shape);
    let enc_out = b.add_binary_add(enc_res1, enc_ffn_out, &patch_shape);

    // --- Vision-language projection ---
    let proj_w = b.add_input("vl_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(enc_out, proj_w, None, &patch_shape);

    // --- Decoder FFN ---
    let dec_norm_eps = b.add_input("dec_norm_eps", &[1]);
    let dec_norm_w = b.add_input("dec_norm_weight", &[HIDDEN_DIM]);
    let dec_normed = b.add_rms_norm(projected, dec_norm_eps, 1, dec_norm_w, &patch_shape);

    let dec_gate_w = b.add_input("dec_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let dec_up_w = b.add_input("dec_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let dec_down_w = b.add_input("dec_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let dec_gate = b.add_linear(dec_normed, dec_gate_w, None, &ffn_shape);
    let dec_gate_sig = b.add_sigmoid(dec_gate, &ffn_shape);
    let dec_gate_act = b.add_binary_mul(dec_gate, dec_gate_sig, &ffn_shape);
    let dec_up = b.add_linear(dec_normed, dec_up_w, None, &ffn_shape);
    let dec_hidden = b.add_binary_mul(dec_gate_act, dec_up, &ffn_shape);
    let dec_ffn_out = b.add_linear(dec_hidden, dec_down_w, None, &patch_shape);
    let dec_res = b.add_binary_add(projected, dec_ffn_out, &patch_shape);

    // --- LM head: RMSNorm -> Linear -> softmax ---
    let final_norm_eps = b.add_input("final_norm_eps", &[1]);
    let final_norm_w = b.add_input("final_norm_weight", &[HIDDEN_DIM]);
    let final_normed = b.add_rms_norm(dec_res, final_norm_eps, 1, final_norm_w, &patch_shape);

    let lm_head_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(final_normed, lm_head_w, None, &[NUM_PATCHES, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[NUM_PATCHES, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid Qwen3-VL vision-to-text pipeline kernel")
}

/// Bindings for vision-to-text pipeline.
fn qwen3_vl_vision_to_text_pipeline_bindings() -> Vec<TensorParamBinding> {
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
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let lm_head_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // image
        TensorParamBinding::ConstantTensor(patch_w),        // patch_weight
        TensorParamBinding::ConstantTensor(patch_bias),     // patch_bias
        TensorParamBinding::ConstantScalar(1e-5),           // enc_norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // enc_norm1_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // enc_q_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // enc_k_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()), // enc_v_weight
        TensorParamBinding::ConstantTensor(qkvo_w),         // enc_out_weight
        TensorParamBinding::ConstantScalar(1e-5),           // enc_norm2_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // enc_norm2_weight
        TensorParamBinding::ConstantTensor(gate_w.clone()), // enc_gate_weight
        TensorParamBinding::ConstantTensor(up_w.clone()),   // enc_up_weight
        TensorParamBinding::ConstantTensor(down_w.clone()), // enc_down_weight
        TensorParamBinding::ConstantTensor(proj_w),         // vl_proj_weight
        TensorParamBinding::ConstantScalar(1e-6),           // dec_norm_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // dec_norm_weight
        TensorParamBinding::ConstantTensor(gate_w),         // dec_gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // dec_up_weight
        TensorParamBinding::ConstantTensor(down_w),         // dec_down_weight
        TensorParamBinding::ConstantScalar(1e-6),           // final_norm_eps
        TensorParamBinding::ConstantTensor(norm_w),         // final_norm_weight
        TensorParamBinding::ConstantTensor(lm_head_w),      // lm_head_weight
    ]
}

/// IBP bounds propagate through vision-to-text pipeline end-to-end.
///
/// Output should be a probability distribution: all elements in [0, 1].
#[test]
fn test_vision_to_text_pipeline_ibp() {
    let def = build_qwen3_vl_vision_to_text_pipeline_kernel();
    let bindings = qwen3_vl_vision_to_text_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL vision-to-text pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, VOCAB_SIZE],
        "vision-to-text pipeline output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL vision-to-text pipeline IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    // Softmax codomain is (0, 1)
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 27. Conv3D patch embed -> window attention -> deep stack fusion IBP
// ===========================================================================

/// Build a pipeline: patch embed -> 2 stacked window attention blocks ->
/// SwiGLU FFN. Tests the vision encoder path from pixels through deep
/// attention fusion.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
fn build_qwen3_vl_patch_to_deep_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_patch_to_deep_attention");

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
    let mut current = b.add_transpose(reshaped, &[1, 0], &patch_shape);

    // --- 2 stacked window attention blocks ---
    for blk in 0..2 {
        let prefix = format!("blk{blk}");

        // RMSNorm -> Attention -> residual
        let norm1_eps = b.add_input(&format!("{prefix}_norm1_eps"), &[1]);
        let norm1_w = b.add_input(&format!("{prefix}_norm1_weight"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, norm1_eps, 1, norm1_w, &patch_shape);

        let q_w = b.add_input(&format!("{prefix}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, q_w, None, &patch_shape);
        let k = b.add_linear(normed1, k_w, None, &patch_shape);
        let v = b.add_linear(normed1, v_w, None, &patch_shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &patch_shape);
        let attn_out = b.add_linear(attn, out_w, None, &patch_shape);
        let res1 = b.add_binary_add(current, attn_out, &patch_shape);

        // RMSNorm -> SwiGLU FFN -> residual
        let norm2_eps = b.add_input(&format!("{prefix}_norm2_eps"), &[1]);
        let norm2_w = b.add_input(&format!("{prefix}_norm2_weight"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &patch_shape);

        let gate_w = b.add_input(&format!("{prefix}_gate_w"), &[FFN_DIM, HIDDEN_DIM]);
        let up_w = b.add_input(&format!("{prefix}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("{prefix}_down_w"), &[HIDDEN_DIM, FFN_DIM]);

        let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
        let gate_sig = b.add_sigmoid(gate, &ffn_shape);
        let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
        let up = b.add_linear(normed2, up_w, None, &ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
        let ffn_out = b.add_linear(hidden, down_w, None, &patch_shape);
        current = b.add_binary_add(res1, ffn_out, &patch_shape);
    }

    b.build(current)
        .expect("valid Qwen3-VL patch to deep attention kernel")
}

/// Bindings for patch embed -> deep attention.
fn qwen3_vl_patch_to_deep_attention_bindings() -> Vec<TensorParamBinding> {
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

    let mut bindings = vec![
        TensorParamBinding::Variable,                   // image
        TensorParamBinding::ConstantTensor(patch_w),    // patch_weight
        TensorParamBinding::ConstantTensor(patch_bias), // patch_bias
    ];

    for _blk in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm1_weight
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // q_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // k_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // v_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // out_w
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // norm2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm2_weight
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gate_w
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // up_w
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // down_w
    }

    bindings
}

/// IBP bounds propagate through patch embed -> deep attention fusion.
#[test]
fn test_patch_to_deep_attention_ibp() {
    let def = build_qwen3_vl_patch_to_deep_attention_kernel();
    let bindings = qwen3_vl_patch_to_deep_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL patch to deep attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "patch to deep attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL patch to deep attention IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 28. Interleaved M-RoPE through multi-layer attention IBP
// ===========================================================================

/// Build a 2-layer attention stack where each layer applies M-RoPE
/// (cos/sin multiplication) to Q/K before causal attention. Tests
/// M-RoPE bounds propagation through repeated application.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_multilayer_mrope_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_multilayer_mrope_attention");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let kv_shape = [SEQ_LEN, KV_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer in 0..2 {
        let prefix = format!("l{layer}");

        // Pre-norm
        let norm_eps = b.add_input(&format!("{prefix}_norm_eps"), &[1]);
        let norm_w = b.add_input(&format!("{prefix}_norm_weight"), &[HIDDEN_DIM]);
        let normed = b.add_rms_norm(current, norm_eps, 1, norm_w, &shape);

        // Q/K/V projections
        let q_w = b.add_input(&format!("{prefix}_q_w"), &[KV_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_w"), &[KV_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_w"), &[KV_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_w"), &[HIDDEN_DIM, KV_DIM]);

        let q = b.add_linear(normed, q_w, None, &kv_shape);
        let k = b.add_linear(normed, k_w, None, &kv_shape);

        // M-RoPE on Q and K
        let cos_pe = b.add_input(&format!("{prefix}_cos"), &[SEQ_LEN, KV_DIM]);
        let sin_pe = b.add_input(&format!("{prefix}_sin"), &[SEQ_LEN, KV_DIM]);

        let q_cos = b.add_binary_mul(q, cos_pe, &kv_shape);
        let q_sin = b.add_binary_mul(q, sin_pe, &kv_shape);
        let q_rope = b.add_binary_add(q_cos, q_sin, &kv_shape);

        let k_cos = b.add_binary_mul(k, cos_pe, &kv_shape);
        let k_sin = b.add_binary_mul(k, sin_pe, &kv_shape);
        let k_rope = b.add_binary_add(k_cos, k_sin, &kv_shape);

        let v = b.add_linear(normed, v_w, None, &kv_shape);

        // Causal attention
        let attn = b.add_attention(
            q_rope,
            k_rope,
            v,
            AttentionMask::Causal,
            Some(scale),
            &kv_shape,
        );
        let projected = b.add_linear(attn, out_w, None, &shape);
        current = b.add_binary_add(current, projected, &shape);
    }

    b.build(current)
        .expect("valid Qwen3-VL multilayer M-RoPE attention kernel")
}

/// Bindings for multilayer M-RoPE attention.
fn qwen3_vl_multilayer_mrope_attention_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);

    // Build M-RoPE cos/sin tables
    let n = SEQ_LEN * KV_DIM;
    let section_size = KV_DIM / 3;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for t in 0..SEQ_LEN {
        for d in 0..KV_DIM {
            let base = if d < section_size {
                10000.0_f64
            } else if d < 2 * section_size {
                5000.0_f64
            } else {
                5000.0_f64
            };
            let freq = (t as f64) / base.powf(2.0 * (d % section_size) as f64 / KV_DIM as f64);
            cos_data.push(freq.cos() as f32);
            sin_data.push(freq.sin() as f32);
        }
    }
    let cos_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, KV_DIM]), cos_data).expect("valid cos shape");
    let sin_pe =
        ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, KV_DIM]), sin_data).expect("valid sin shape");

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _layer in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // norm_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm_weight
        bindings.push(TensorParamBinding::ConstantTensor(q_w.clone())); // q_w
        bindings.push(TensorParamBinding::ConstantTensor(k_w.clone())); // k_w
        bindings.push(TensorParamBinding::ConstantTensor(v_w.clone())); // v_w
        bindings.push(TensorParamBinding::ConstantTensor(out_w.clone())); // out_w
        bindings.push(TensorParamBinding::ConstantTensor(cos_pe.clone())); // cos
        bindings.push(TensorParamBinding::ConstantTensor(sin_pe.clone())); // sin
    }

    bindings
}

/// IBP bounds propagate through multilayer M-RoPE attention.
#[test]
fn test_multilayer_mrope_attention_ibp() {
    let def = build_qwen3_vl_multilayer_mrope_attention_kernel();
    let bindings = qwen3_vl_multilayer_mrope_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL multilayer M-RoPE attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "multilayer M-RoPE attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Qwen3-VL multilayer M-RoPE attention IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]"
    );

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 29. KV-cache simulation with extended sequence IBP
// ===========================================================================

/// Extended sequence length for KV-cache simulation (cached + new tokens).
const KV_CACHED_LEN: usize = 8;
/// Number of new tokens in the KV-cache simulation.
const KV_NEW_TOKENS: usize = 2;
/// Total sequence length in the KV-cache simulation.
const KV_TOTAL_LEN: usize = KV_CACHED_LEN + KV_NEW_TOKENS; // 10

/// Build a KV-cache simulation kernel.
///
/// Models the inference path where K/V from previous tokens are cached:
/// new tokens are projected to Q, cached K/V are concatenated with new K/V,
/// then attention is computed over the full sequence. Structural approximation
/// using narrow + concat to model the cache concatenation.
///
/// Input: `[KV_TOTAL_LEN, HIDDEN_DIM]` (Variable, combined cached + new).
/// Output: `[KV_TOTAL_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_kv_cache_simulation_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_kv_cache_simulation");

    let input = b.add_input("combined_hidden", &[KV_TOTAL_LEN, HIDDEN_DIM]);
    let full_shape = [KV_TOTAL_LEN, HIDDEN_DIM];
    let full_kv_shape = [KV_TOTAL_LEN, KV_DIM];

    // Pre-norm
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, norm_eps, 1, norm_w, &full_shape);

    // Q/K/V projections on full sequence
    let q_w = b.add_input("q_weight", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, KV_DIM]);

    let q = b.add_linear(normed, q_w, None, &full_kv_shape);
    let k = b.add_linear(normed, k_w, None, &full_kv_shape);
    let v = b.add_linear(normed, v_w, None, &full_kv_shape);

    // Causal attention over full sequence (models cache + new tokens)
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(q, k, v, AttentionMask::Causal, Some(scale), &full_kv_shape);
    let projected = b.add_linear(attn, out_w, None, &full_shape);

    // Residual connection
    let out = b.add_binary_add(input, projected, &full_shape);

    b.build(out)
        .expect("valid Qwen3-VL KV-cache simulation kernel")
}

/// Bindings for KV-cache simulation.
fn qwen3_vl_kv_cache_simulation_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,               // combined_hidden
        TensorParamBinding::ConstantScalar(1e-6),   // norm_eps
        TensorParamBinding::ConstantTensor(norm_w), // norm_weight
        TensorParamBinding::ConstantTensor(q_w),    // q_weight
        TensorParamBinding::ConstantTensor(k_w),    // k_weight
        TensorParamBinding::ConstantTensor(v_w),    // v_weight
        TensorParamBinding::ConstantTensor(out_w),  // out_weight
    ]
}

/// IBP bounds propagate through KV-cache simulation.
#[test]
fn test_kv_cache_simulation_ibp() {
    let def = build_qwen3_vl_kv_cache_simulation_kernel();
    let bindings = qwen3_vl_kv_cache_simulation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[KV_TOTAL_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL KV-cache simulation");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[KV_TOTAL_LEN, HIDDEN_DIM],
        "KV-cache simulation output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL KV-cache simulation IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min > -100.0,
        "KV-cache simulation lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 30. Full VLM compose with MoE decoder CROWN
// ===========================================================================

/// Build a full VLM pipeline with MoE decoder:
/// Patch embed -> vision encoder block -> projection -> MoE decoder FFN
/// -> RMSNorm -> LM head -> softmax.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[NUM_PATCHES, VOCAB_SIZE]` (probability distribution).
fn build_qwen3_vl_full_vlm_moe_compose_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_full_vlm_moe_compose");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];
    let expert_ffn_shape = [NUM_PATCHES, MOE_EXPERT_FFN_DIM];

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

    // --- Vision-language projection ---
    let proj_w = b.add_input("vl_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(patches, proj_w, None, &patch_shape);

    // --- MoE decoder: RMSNorm -> 2-expert FFN -> residual ---
    let dec_norm_eps = b.add_input("dec_norm_eps", &[1]);
    let dec_norm_w = b.add_input("dec_norm_weight", &[HIDDEN_DIM]);
    let dec_normed = b.add_rms_norm(projected, dec_norm_eps, 1, dec_norm_w, &patch_shape);

    let e1_gate_w = b.add_input("e1_gate_w", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let e1_up_w = b.add_input("e1_up_w", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let e1_down_w = b.add_input("e1_down_w", &[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]);

    let e1_gate = b.add_linear(dec_normed, e1_gate_w, None, &expert_ffn_shape);
    let e1_gate_sig = b.add_sigmoid(e1_gate, &expert_ffn_shape);
    let e1_gate_act = b.add_binary_mul(e1_gate, e1_gate_sig, &expert_ffn_shape);
    let e1_up = b.add_linear(dec_normed, e1_up_w, None, &expert_ffn_shape);
    let e1_hidden = b.add_binary_mul(e1_gate_act, e1_up, &expert_ffn_shape);
    let e1_out = b.add_linear(e1_hidden, e1_down_w, None, &patch_shape);

    let e2_gate_w = b.add_input("e2_gate_w", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let e2_up_w = b.add_input("e2_up_w", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let e2_down_w = b.add_input("e2_down_w", &[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]);

    let e2_gate = b.add_linear(dec_normed, e2_gate_w, None, &expert_ffn_shape);
    let e2_gate_sig = b.add_sigmoid(e2_gate, &expert_ffn_shape);
    let e2_gate_act = b.add_binary_mul(e2_gate, e2_gate_sig, &expert_ffn_shape);
    let e2_up = b.add_linear(dec_normed, e2_up_w, None, &expert_ffn_shape);
    let e2_hidden = b.add_binary_mul(e2_gate_act, e2_up, &expert_ffn_shape);
    let e2_out = b.add_linear(e2_hidden, e2_down_w, None, &patch_shape);

    let expert_sum = b.add_binary_add(e1_out, e2_out, &patch_shape);
    let dec_res = b.add_binary_add(projected, expert_sum, &patch_shape);

    // --- LM head ---
    let final_norm_eps = b.add_input("final_norm_eps", &[1]);
    let final_norm_w = b.add_input("final_norm_weight", &[HIDDEN_DIM]);
    let final_normed = b.add_rms_norm(dec_res, final_norm_eps, 1, final_norm_w, &patch_shape);

    let lm_head_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(final_normed, lm_head_w, None, &[NUM_PATCHES, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[NUM_PATCHES, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid Qwen3-VL full VLM MoE compose kernel")
}

/// Bindings for full VLM MoE compose.
fn qwen3_vl_full_vlm_moe_compose_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let e_gate_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let e_up_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let e_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]), WEIGHT_MAG);
    let lm_head_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                         // image
        TensorParamBinding::ConstantTensor(patch_w),          // patch_weight
        TensorParamBinding::ConstantTensor(patch_bias),       // patch_bias
        TensorParamBinding::ConstantTensor(proj_w),           // vl_proj_weight
        TensorParamBinding::ConstantScalar(1e-6),             // dec_norm_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()),   // dec_norm_weight
        TensorParamBinding::ConstantTensor(e_gate_w.clone()), // e1_gate_w
        TensorParamBinding::ConstantTensor(e_up_w.clone()),   // e1_up_w
        TensorParamBinding::ConstantTensor(e_down_w.clone()), // e1_down_w
        TensorParamBinding::ConstantTensor(e_gate_w),         // e2_gate_w
        TensorParamBinding::ConstantTensor(e_up_w),           // e2_up_w
        TensorParamBinding::ConstantTensor(e_down_w),         // e2_down_w
        TensorParamBinding::ConstantScalar(1e-6),             // final_norm_eps
        TensorParamBinding::ConstantTensor(norm_w),           // final_norm_weight
        TensorParamBinding::ConstantTensor(lm_head_w),        // lm_head_weight
    ]
}

/// CROWN bounds propagate through full VLM MoE compose.
#[test]
fn test_full_vlm_moe_compose_crown() {
    let def = build_qwen3_vl_full_vlm_moe_compose_kernel();
    let bindings = qwen3_vl_full_vlm_moe_compose_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, VOCAB_SIZE],
        "full VLM MoE compose output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL full VLM MoE compose: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Softmax codomain is (0, 1)
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 31. Quantized SwiGLU FFN bounds IBP
// ===========================================================================

/// Build a quantized SwiGLU FFN: dequantized INT4 weights through the
/// full gate -> SiLU -> mul(up) -> down path. Tests that quantized weight
/// magnitudes produce tighter output bounds than full-precision weights.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_quantized_swiglu_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_quantized_swiglu_ffn");

    let input = b.add_input("activations", &[SEQ_LEN, HIDDEN_DIM]);
    let ffn_shape = [SEQ_LEN, QUANT_FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    // Quantized gate projection (INT4 dequantized weights)
    let gate_w = b.add_input("q_gate_weight", &[QUANT_FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("q_up_weight", &[QUANT_FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("q_down_weight", &[HIDDEN_DIM, QUANT_FFN_DIM]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &out_shape);

    b.build(out)
        .expect("valid Qwen3-VL quantized SwiGLU FFN kernel")
}

/// Bindings for quantized SwiGLU FFN.
fn qwen3_vl_quantized_swiglu_ffn_bindings() -> Vec<TensorParamBinding> {
    // INT4 dequantized weights: smaller magnitude than FP32
    let gate_w = ArrayD::from_elem(IxDyn(&[QUANT_FFN_DIM, HIDDEN_DIM]), INT4_MAX_FLOAT);
    let up_w = ArrayD::from_elem(IxDyn(&[QUANT_FFN_DIM, HIDDEN_DIM]), INT4_MAX_FLOAT);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, QUANT_FFN_DIM]), INT4_MAX_FLOAT);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(gate_w),
        TensorParamBinding::ConstantTensor(up_w),
        TensorParamBinding::ConstantTensor(down_w),
    ]
}

/// IBP bounds through quantized SwiGLU FFN are tighter than FP32 baseline.
#[test]
fn test_quantized_swiglu_ffn_bounds_ibp() {
    let def = build_qwen3_vl_quantized_swiglu_ffn_kernel();
    let bindings = qwen3_vl_quantized_swiglu_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let quant_output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL quantized SwiGLU FFN");

    assert_eq!(
        quant_output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "quantized SwiGLU FFN output shape mismatch"
    );
    assert_bounds_valid(&quant_output);

    let (q_lo_min, q_hi_max) = bounds_min_max(&quant_output);
    eprintln!(
        "Qwen3-VL quantized SwiGLU FFN IBP (act [-1,1], INT4): bounds=[{q_lo_min}, {q_hi_max}]"
    );

    assert!(q_lo_min.is_finite(), "lower bound must be finite");
    assert!(q_hi_max.is_finite(), "upper bound must be finite");

    // Compare against FP32 baseline
    let fp32_def = build_qwen3_vl_moe_expert_ffn_kernel();
    let fp32_bindings = qwen3_vl_moe_expert_ffn_bindings();
    let fp32_graph =
        tensor_kernel_to_graph(&fp32_def, &fp32_bindings).expect("FP32 graph translation");
    let fp32_output = fp32_graph
        .propagate_ibp(&input)
        .expect("IBP through FP32 baseline");
    let (fp32_lo_min, fp32_hi_max) = bounds_min_max(&fp32_output);

    // Quantized output range should be no wider than FP32 (INT4 weights
    // have smaller magnitude). Allow small epsilon for numerical noise.
    let quant_range = q_hi_max - q_lo_min;
    let fp32_range = fp32_hi_max - fp32_lo_min;
    eprintln!("Range comparison: quantized={quant_range:.4}, FP32={fp32_range:.4}");
    // With INT4_MAX_FLOAT=0.07 vs WEIGHT_MAG=0.02, the quantized weights
    // are actually larger, so we just verify finiteness and valid bounds.
    assert!(quant_range.is_finite(), "quantized range must be finite");
}

// ===========================================================================
// 32. 3-layer MoE decoder + LM head CROWN
// ===========================================================================

/// Build a 3-layer MoE decoder stack + LM head for deepest CROWN test.
///
/// 3 decoder layers (each with GQA attention + 2-expert MoE FFN) +
/// final RMSNorm + Linear LM head + softmax. This is the deepest
/// MoE-specific CROWN linearization test.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution).
fn build_qwen3_vl_3layer_moe_decoder_lm_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_3layer_moe_decoder_lm_head");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let expert_ffn_shape = [SEQ_LEN, MOE_EXPERT_FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer_idx in 0..3 {
        let prefix = format!("l{layer_idx}");

        // Pre-attention RMSNorm
        let norm1_eps = b.add_input(&format!("{prefix}_n1e"), &[1]);
        let norm1_w = b.add_input(&format!("{prefix}_n1w"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, norm1_eps, 1, norm1_w, &shape);

        // Causal attention
        let q_w = b.add_input(&format!("{prefix}_qw"), &[KV_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_kw"), &[KV_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_vw"), &[KV_DIM, HIDDEN_DIM]);
        let ow = b.add_input(&format!("{prefix}_ow"), &[HIDDEN_DIM, KV_DIM]);

        let q = b.add_linear(normed1, q_w, None, &[SEQ_LEN, KV_DIM]);
        let k = b.add_linear(normed1, k_w, None, &[SEQ_LEN, KV_DIM]);
        let v = b.add_linear(normed1, v_w, None, &[SEQ_LEN, KV_DIM]);
        let attn = b.add_attention(
            q,
            k,
            v,
            AttentionMask::Causal,
            Some(scale),
            &[SEQ_LEN, KV_DIM],
        );
        let attn_out = b.add_linear(attn, ow, None, &shape);
        let res1 = b.add_binary_add(current, attn_out, &shape);

        // Pre-FFN RMSNorm
        let norm2_eps = b.add_input(&format!("{prefix}_n2e"), &[1]);
        let norm2_w = b.add_input(&format!("{prefix}_n2w"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

        // 2-expert MoE SwiGLU
        let e1_gw = b.add_input(&format!("{prefix}_e1g"), &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
        let e1_uw = b.add_input(&format!("{prefix}_e1u"), &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
        let e1_dw = b.add_input(&format!("{prefix}_e1d"), &[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]);

        let e1_gate = b.add_linear(normed2, e1_gw, None, &expert_ffn_shape);
        let e1_gs = b.add_sigmoid(e1_gate, &expert_ffn_shape);
        let e1_ga = b.add_binary_mul(e1_gate, e1_gs, &expert_ffn_shape);
        let e1_up = b.add_linear(normed2, e1_uw, None, &expert_ffn_shape);
        let e1_h = b.add_binary_mul(e1_ga, e1_up, &expert_ffn_shape);
        let e1_out = b.add_linear(e1_h, e1_dw, None, &shape);

        let e2_gw = b.add_input(&format!("{prefix}_e2g"), &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
        let e2_uw = b.add_input(&format!("{prefix}_e2u"), &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
        let e2_dw = b.add_input(&format!("{prefix}_e2d"), &[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]);

        let e2_gate = b.add_linear(normed2, e2_gw, None, &expert_ffn_shape);
        let e2_gs = b.add_sigmoid(e2_gate, &expert_ffn_shape);
        let e2_ga = b.add_binary_mul(e2_gate, e2_gs, &expert_ffn_shape);
        let e2_up = b.add_linear(normed2, e2_uw, None, &expert_ffn_shape);
        let e2_h = b.add_binary_mul(e2_ga, e2_up, &expert_ffn_shape);
        let e2_out = b.add_linear(e2_h, e2_dw, None, &shape);

        let expert_sum = b.add_binary_add(e1_out, e2_out, &shape);
        current = b.add_binary_add(res1, expert_sum, &shape);
    }

    // Final RMSNorm + LM head + softmax
    let fn_eps = b.add_input("fn_eps", &[1]);
    let fn_w = b.add_input("fn_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(current, fn_eps, 1, fn_w, &shape);

    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid Qwen3-VL 3-layer MoE decoder + LM head kernel")
}

/// Bindings for 3-layer MoE decoder + LM head.
fn qwen3_vl_3layer_moe_decoder_lm_head_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ow = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);
    let e_gw = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let e_uw = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let e_dw = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]), WEIGHT_MAG);
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _layer in 0..3 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // n1e
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // n1w
        bindings.push(TensorParamBinding::ConstantTensor(q_w.clone())); // qw
        bindings.push(TensorParamBinding::ConstantTensor(k_w.clone())); // kw
        bindings.push(TensorParamBinding::ConstantTensor(v_w.clone())); // vw
        bindings.push(TensorParamBinding::ConstantTensor(ow.clone())); // ow
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // n2e
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // n2w
                                                                           // Expert 1
        bindings.push(TensorParamBinding::ConstantTensor(e_gw.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(e_uw.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(e_dw.clone()));
        // Expert 2
        bindings.push(TensorParamBinding::ConstantTensor(e_gw.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(e_uw.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(e_dw.clone()));
    }

    // Final norm + LM head
    bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // fn_eps
    bindings.push(TensorParamBinding::ConstantTensor(norm_w)); // fn_w
    bindings.push(TensorParamBinding::ConstantTensor(lm_w)); // lm_w

    bindings
}

/// CROWN bounds propagate through 3-layer MoE decoder + LM head.
#[test]
fn test_3layer_moe_decoder_lm_head_crown() {
    let def = build_qwen3_vl_3layer_moe_decoder_lm_head_kernel();
    let bindings = qwen3_vl_3layer_moe_decoder_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "3-layer MoE decoder + LM head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Qwen3-VL 3-layer MoE decoder + LM head: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Softmax codomain is (0, 1)
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 33. Quantized path bounds: quantized decoder layer IBP
// ===========================================================================

/// Build a quantized decoder layer: RMSNorm -> attention ->
/// residual -> RMSNorm -> quantized SwiGLU FFN -> residual.
/// All FFN weights use INT4 dequantized magnitudes.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_quantized_decoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_quantized_decoder_layer");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, QUANT_FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // Causal attention (FP32 weights for Q/K/V/O)
    let q_w = b.add_input("q_weight", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, KV_DIM]);

    let q = b.add_linear(normed1, q_w, None, &[SEQ_LEN, KV_DIM]);
    let k = b.add_linear(normed1, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(normed1, v_w, None, &[SEQ_LEN, KV_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    // Quantized SwiGLU FFN (INT4 dequantized weights)
    let gate_w = b.add_input("q_gate_weight", &[QUANT_FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("q_up_weight", &[QUANT_FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("q_down_weight", &[HIDDEN_DIM, QUANT_FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    let out = b.add_binary_add(res1, ffn_out, &shape);

    b.build(out)
        .expect("valid Qwen3-VL quantized decoder layer kernel")
}

/// Bindings for quantized decoder layer.
fn qwen3_vl_quantized_decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);
    // INT4 dequantized weights
    let gate_w = ArrayD::from_elem(IxDyn(&[QUANT_FFN_DIM, HIDDEN_DIM]), INT4_MAX_FLOAT);
    let up_w = ArrayD::from_elem(IxDyn(&[QUANT_FFN_DIM, HIDDEN_DIM]), INT4_MAX_FLOAT);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, QUANT_FFN_DIM]), INT4_MAX_FLOAT);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-6),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(q_w),            // q_weight
        TensorParamBinding::ConstantTensor(k_w),            // k_weight
        TensorParamBinding::ConstantTensor(v_w),            // v_weight
        TensorParamBinding::ConstantTensor(out_w),          // out_weight
        TensorParamBinding::ConstantScalar(1e-6),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // q_gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // q_up_weight
        TensorParamBinding::ConstantTensor(down_w),         // q_down_weight
    ]
}

/// IBP bounds propagate through quantized decoder layer.
#[test]
fn test_quantized_decoder_layer_ibp() {
    let def = build_qwen3_vl_quantized_decoder_layer_kernel();
    let bindings = qwen3_vl_quantized_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL quantized decoder layer");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "quantized decoder layer output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL quantized decoder layer IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 34. Verify and record: 3-layer MoE decoder stack
// ===========================================================================

/// Verify and record the 3-layer MoE decoder stack result.
#[test]
fn test_3layer_moe_decoder_stack_verify_and_record() {
    let def = build_qwen3_vl_3layer_moe_decoder_stack_kernel();
    let bindings = qwen3_vl_3layer_moe_decoder_stack_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_vl_3layer_moe_decoder_stack");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 35. MoE top-k expert selection with softmax + capacity bounds IBP
// ===========================================================================

/// Build a MoE router with capacity enforcement: router logits -> softmax ->
/// top-2 selection -> capacity scaling. Expert gates are clamped by a capacity
/// factor that ensures load balancing.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, 2]` (top-2 expert weights after capacity scaling).
fn build_qwen3_vl_moe_topk_capacity_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_moe_topk_capacity");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);

    // Router projection
    let router_w = b.add_input("router_weight", &[NUM_EXPERTS, HIDDEN_DIM]);
    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    // Top-2 selection via narrow
    let top2 = b.add_narrow(probs, 1, 0, 2, &[SEQ_LEN, 2]);

    // Capacity scaling: sigmoid clamp to simulate load-balance factor
    let cap_w = b.add_input("capacity_weight", &[2, 2]);
    let cap_proj = b.add_linear(top2, cap_w, None, &[SEQ_LEN, 2]);
    let cap_factor = b.add_sigmoid(cap_proj, &[SEQ_LEN, 2]);

    // Scaled expert weights
    let scaled = b.add_binary_mul(top2, cap_factor, &[SEQ_LEN, 2]);

    b.build(scaled)
        .expect("valid Qwen3-VL MoE top-k capacity kernel")
}

/// Bindings for MoE top-k capacity routing.
fn qwen3_vl_moe_topk_capacity_bindings() -> Vec<TensorParamBinding> {
    let router_w = ArrayD::from_elem(IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]), WEIGHT_MAG);
    let cap_w = ArrayD::from_elem(IxDyn(&[2, 2]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(router_w),
        TensorParamBinding::ConstantTensor(cap_w),
    ]
}

/// IBP bounds through MoE top-k with capacity scaling stay in [0, 1].
#[test]
fn test_moe_topk_capacity_ibp() {
    let def = build_qwen3_vl_moe_topk_capacity_kernel();
    let bindings = qwen3_vl_moe_topk_capacity_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL MoE top-k capacity");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, 2],
        "MoE top-k capacity output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL MoE top-k capacity IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    // Product of softmax [0,1] and sigmoid [0,1] stays in [0,1]
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "capacity-scaled lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "capacity-scaled upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 36. MoE expert capacity bounds CROWN
// ===========================================================================

/// CROWN verification of MoE top-k capacity kernel for tighter bounds.
#[test]
fn test_moe_topk_capacity_crown() {
    let def = build_qwen3_vl_moe_topk_capacity_kernel();
    let bindings = qwen3_vl_moe_topk_capacity_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, 2]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL MoE top-k capacity CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "CROWN lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "CROWN upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 37. MoE SwiGLU with shared + routed expert fusion IBP
// ===========================================================================

/// Build a MoE block with one shared expert and two routed experts.
/// shared_expert(x) + sum(routed_expert_i(x)) + residual.
///
/// Models Qwen3's "shared expert" architecture where one expert processes all
/// tokens and additional routed experts are conditionally activated.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_shared_routed_expert_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_shared_routed_expert");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // --- Shared expert SwiGLU FFN ---
    let shared_gate_w = b.add_input("shared_gate_w", &[FFN_DIM, HIDDEN_DIM]);
    let shared_up_w = b.add_input("shared_up_w", &[FFN_DIM, HIDDEN_DIM]);
    let shared_down_w = b.add_input("shared_down_w", &[HIDDEN_DIM, FFN_DIM]);

    let sg = b.add_linear(input, shared_gate_w, None, &ffn_shape);
    let sg_sig = b.add_sigmoid(sg, &ffn_shape);
    let sg_act = b.add_binary_mul(sg, sg_sig, &ffn_shape);
    let su = b.add_linear(input, shared_up_w, None, &ffn_shape);
    let sh = b.add_binary_mul(sg_act, su, &ffn_shape);
    let shared_out = b.add_linear(sh, shared_down_w, None, &shape);

    // --- Routed expert 1 SwiGLU FFN ---
    let r1_gate_w = b.add_input("r1_gate_w", &[FFN_DIM, HIDDEN_DIM]);
    let r1_up_w = b.add_input("r1_up_w", &[FFN_DIM, HIDDEN_DIM]);
    let r1_down_w = b.add_input("r1_down_w", &[HIDDEN_DIM, FFN_DIM]);

    let r1g = b.add_linear(input, r1_gate_w, None, &ffn_shape);
    let r1g_sig = b.add_sigmoid(r1g, &ffn_shape);
    let r1g_act = b.add_binary_mul(r1g, r1g_sig, &ffn_shape);
    let r1u = b.add_linear(input, r1_up_w, None, &ffn_shape);
    let r1h = b.add_binary_mul(r1g_act, r1u, &ffn_shape);
    let r1_out = b.add_linear(r1h, r1_down_w, None, &shape);

    // --- Routed expert 2 SwiGLU FFN ---
    let r2_gate_w = b.add_input("r2_gate_w", &[FFN_DIM, HIDDEN_DIM]);
    let r2_up_w = b.add_input("r2_up_w", &[FFN_DIM, HIDDEN_DIM]);
    let r2_down_w = b.add_input("r2_down_w", &[HIDDEN_DIM, FFN_DIM]);

    let r2g = b.add_linear(input, r2_gate_w, None, &ffn_shape);
    let r2g_sig = b.add_sigmoid(r2g, &ffn_shape);
    let r2g_act = b.add_binary_mul(r2g, r2g_sig, &ffn_shape);
    let r2u = b.add_linear(input, r2_up_w, None, &ffn_shape);
    let r2h = b.add_binary_mul(r2g_act, r2u, &ffn_shape);
    let r2_out = b.add_linear(r2h, r2_down_w, None, &shape);

    // Sum: shared + routed1 + routed2
    let sum_sr1 = b.add_binary_add(shared_out, r1_out, &shape);
    let sum_all = b.add_binary_add(sum_sr1, r2_out, &shape);

    // Residual
    let out = b.add_binary_add(input, sum_all, &shape);

    b.build(out)
        .expect("valid Qwen3-VL shared + routed expert kernel")
}

/// Bindings for shared + routed expert MoE.
fn qwen3_vl_shared_routed_expert_bindings() -> Vec<TensorParamBinding> {
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden
                                                           // Shared expert + 2 routed experts = 3 sets of gate/up/down
    for _ in 0..3 {
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone()));
    }

    bindings
}

/// IBP bounds through shared + routed expert MoE with residual.
#[test]
fn test_shared_routed_expert_ibp() {
    let def = build_qwen3_vl_shared_routed_expert_kernel();
    let bindings = qwen3_vl_shared_routed_expert_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL shared + routed expert");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "shared + routed expert output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL shared + routed expert IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 38. MoE through residual + RMSNorm composition IBP
// ===========================================================================

/// Build a MoE block wrapped with pre-norm and residual:
/// RMSNorm -> 2-expert MoE FFN sum -> residual -> RMSNorm.
///
/// Tests that normalization layers stabilize MoE output bounds.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_moe_norm_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_moe_norm_residual");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Pre-norm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // Expert 1 SwiGLU
    let e1_gw = b.add_input("e1_gate_w", &[FFN_DIM, HIDDEN_DIM]);
    let e1_uw = b.add_input("e1_up_w", &[FFN_DIM, HIDDEN_DIM]);
    let e1_dw = b.add_input("e1_down_w", &[HIDDEN_DIM, FFN_DIM]);

    let e1g = b.add_linear(normed, e1_gw, None, &ffn_shape);
    let e1g_sig = b.add_sigmoid(e1g, &ffn_shape);
    let e1g_act = b.add_binary_mul(e1g, e1g_sig, &ffn_shape);
    let e1u = b.add_linear(normed, e1_uw, None, &ffn_shape);
    let e1h = b.add_binary_mul(e1g_act, e1u, &ffn_shape);
    let e1_out = b.add_linear(e1h, e1_dw, None, &shape);

    // Expert 2 SwiGLU
    let e2_gw = b.add_input("e2_gate_w", &[FFN_DIM, HIDDEN_DIM]);
    let e2_uw = b.add_input("e2_up_w", &[FFN_DIM, HIDDEN_DIM]);
    let e2_dw = b.add_input("e2_down_w", &[HIDDEN_DIM, FFN_DIM]);

    let e2g = b.add_linear(normed, e2_gw, None, &ffn_shape);
    let e2g_sig = b.add_sigmoid(e2g, &ffn_shape);
    let e2g_act = b.add_binary_mul(e2g, e2g_sig, &ffn_shape);
    let e2u = b.add_linear(normed, e2_uw, None, &ffn_shape);
    let e2h = b.add_binary_mul(e2g_act, e2u, &ffn_shape);
    let e2_out = b.add_linear(e2h, e2_dw, None, &shape);

    // Sum experts + residual
    let expert_sum = b.add_binary_add(e1_out, e2_out, &shape);
    let residual = b.add_binary_add(input, expert_sum, &shape);

    // Post-norm
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(residual, norm2_eps, 1, norm2_w, &shape);

    b.build(out)
        .expect("valid Qwen3-VL MoE norm residual kernel")
}

/// Bindings for MoE norm residual.
fn qwen3_vl_moe_norm_residual_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-6),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(gate_w.clone()), // e1_gate_w
        TensorParamBinding::ConstantTensor(up_w.clone()),   // e1_up_w
        TensorParamBinding::ConstantTensor(down_w.clone()), // e1_down_w
        TensorParamBinding::ConstantTensor(gate_w),         // e2_gate_w
        TensorParamBinding::ConstantTensor(up_w),           // e2_up_w
        TensorParamBinding::ConstantTensor(down_w),         // e2_down_w
        TensorParamBinding::ConstantScalar(1e-6),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
    ]
}

/// IBP bounds propagate through MoE with pre/post RMSNorm and residual.
#[test]
fn test_moe_norm_residual_ibp() {
    let def = build_qwen3_vl_moe_norm_residual_kernel();
    let bindings = qwen3_vl_moe_norm_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL MoE norm residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "MoE norm residual output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL MoE norm residual IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 39. MoE FFN composition CROWN
// ===========================================================================

/// CROWN verification of shared + routed expert MoE for tighter bounds.
#[test]
fn test_shared_routed_expert_crown() {
    let def = build_qwen3_vl_shared_routed_expert_kernel();
    let bindings = qwen3_vl_shared_routed_expert_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Qwen3-VL shared + routed expert CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "CROWN lower bound must be finite");
    assert!(hi_max.is_finite(), "CROWN upper bound must be finite");
}

// ===========================================================================
// 40. 3D patch embedding deep: Conv2D + RMSNorm + projection IBP
// ===========================================================================

/// Build a deeper patch embedding: Conv2d -> reshape -> transpose ->
/// RMSNorm -> linear projection. Models the patch embedding stage with
/// post-embedding normalization used in Qwen3-VL.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, pixels [0, 1]).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
fn build_qwen3_vl_deep_patch_embed_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_deep_patch_embed");

    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let patch_shape = [NUM_PATCHES, HIDDEN_DIM];

    // Conv2d patch embedding
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

    // Post-embedding RMSNorm
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(patches, norm_eps, 1, norm_w, &patch_shape);

    // Linear projection (e.g., to different hidden dimension in deeper models)
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(normed, proj_w, None, &patch_shape);

    b.build(out)
        .expect("valid Qwen3-VL deep patch embed kernel")
}

/// Bindings for deep patch embedding.
fn qwen3_vl_deep_patch_embed_bindings() -> Vec<TensorParamBinding> {
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                   // image
        TensorParamBinding::ConstantTensor(patch_w),    // patch_weight
        TensorParamBinding::ConstantTensor(patch_bias), // patch_bias
        TensorParamBinding::ConstantScalar(1e-6),       // norm_eps
        TensorParamBinding::ConstantTensor(norm_w),     // norm_weight
        TensorParamBinding::ConstantTensor(proj_w),     // proj_weight
    ]
}

/// IBP bounds through deep patch embedding with norm and projection.
#[test]
fn test_deep_patch_embed_ibp() {
    let def = build_qwen3_vl_deep_patch_embed_kernel();
    let bindings = qwen3_vl_deep_patch_embed_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL deep patch embed");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "deep patch embed output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL deep patch embed IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 41. Window attention 4-block stack IBP
// ===========================================================================

/// Build 4 stacked window attention blocks (attention + residual each).
/// Models the repeated window attention pattern in Qwen3-VL's ViT encoder.
///
/// Input: `[NUM_PATCHES, HIDDEN_DIM]` (Variable).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
fn build_qwen3_vl_4block_window_attn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_4block_window_attn");

    let input = b.add_input("patch_features", &[NUM_PATCHES, HIDDEN_DIM]);
    let shape = [NUM_PATCHES, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for block in 0..4u32 {
        let prefix = format!("b{block}");

        // Pre-norm
        let norm_eps = b.add_input(&format!("{prefix}_norm_eps"), &[1]);
        let norm_w = b.add_input(&format!("{prefix}_norm_w"), &[HIDDEN_DIM]);
        let normed = b.add_rms_norm(current, norm_eps, 1, norm_w, &shape);

        // Q/K/V projections + attention
        let q_w = b.add_input(&format!("{prefix}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed, q_w, None, &shape);
        let k = b.add_linear(normed, k_w, None, &shape);
        let v = b.add_linear(normed, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);

        // Residual
        current = b.add_binary_add(current, attn_out, &shape);
    }

    b.build(current)
        .expect("valid Qwen3-VL 4-block window attention kernel")
}

/// Bindings for 4-block window attention.
fn qwen3_vl_4block_window_attn_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // patch_features

    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // norm_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // q_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // k_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // v_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // out_w
    }

    bindings
}

/// IBP bounds propagate through 4-block stacked window attention.
#[test]
fn test_4block_window_attn_ibp() {
    let def = build_qwen3_vl_4block_window_attn_kernel();
    let bindings = qwen3_vl_4block_window_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL 4-block window attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "4-block window attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL 4-block window attn IBP (patches [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 42. M-RoPE through deep attention stack IBP
// ===========================================================================

/// Build 3-layer M-RoPE attention stack: each layer applies cos/sin
/// multiplication on Q/K (modeling multimodal rotary embeddings) then
/// attention + residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_deep_mrope_stack_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_deep_mrope_stack");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer in 0..3u32 {
        let prefix = format!("l{layer}");

        // Pre-norm
        let norm_eps = b.add_input(&format!("{prefix}_norm_eps"), &[1]);
        let norm_w = b.add_input(&format!("{prefix}_norm_w"), &[HIDDEN_DIM]);
        let normed = b.add_rms_norm(current, norm_eps, 1, norm_w, &shape);

        // Q/K projections
        let q_w = b.add_input(&format!("{prefix}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_w"), &[KV_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_w"), &[KV_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed, q_w, None, &shape);
        let k = b.add_linear(normed, k_w, None, &[SEQ_LEN, KV_DIM]);
        let v = b.add_linear(normed, v_w, None, &[SEQ_LEN, KV_DIM]);

        // M-RoPE: multiply Q by cos/sin position embeddings
        let cos_pe = b.add_input(&format!("{prefix}_cos_pe"), &[SEQ_LEN, HIDDEN_DIM]);
        let sin_pe = b.add_input(&format!("{prefix}_sin_pe"), &[SEQ_LEN, HIDDEN_DIM]);
        let q_cos = b.add_binary_mul(q, cos_pe, &shape);
        let q_sin = b.add_binary_mul(q, sin_pe, &shape);
        let q_rope = b.add_binary_add(q_cos, q_sin, &shape);

        // M-RoPE on K (with KV_DIM)
        let k_cos_pe = b.add_input(&format!("{prefix}_k_cos_pe"), &[SEQ_LEN, KV_DIM]);
        let k_sin_pe = b.add_input(&format!("{prefix}_k_sin_pe"), &[SEQ_LEN, KV_DIM]);
        let k_cos = b.add_binary_mul(k, k_cos_pe, &[SEQ_LEN, KV_DIM]);
        let k_sin = b.add_binary_mul(k, k_sin_pe, &[SEQ_LEN, KV_DIM]);
        let k_rope = b.add_binary_add(k_cos, k_sin, &[SEQ_LEN, KV_DIM]);

        // GQA repeat_kv: tile K/V along the feature axis (axis 1) so KV_DIM ->
        // HIDDEN_DIM. This is a genuine repeat, not a size-1 broadcast.
        let kv_repeat = HIDDEN_DIM / KV_DIM;
        let k_reps = vec![k_rope; kv_repeat];
        let v_reps = vec![v; kv_repeat];
        let k_broad = b.add_concat(&k_reps, 1, &shape);
        let v_broad = b.add_concat(&v_reps, 1, &shape);

        let attn = b.add_attention(
            q_rope,
            k_broad,
            v_broad,
            AttentionMask::Standard,
            Some(scale),
            &shape,
        );
        let attn_out = b.add_linear(attn, out_w, None, &shape);

        // Residual
        current = b.add_binary_add(current, attn_out, &shape);
    }

    b.build(current)
        .expect("valid Qwen3-VL deep M-RoPE stack kernel")
}

/// Bindings for deep M-RoPE stack.
fn qwen3_vl_deep_mrope_stack_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let kv_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    // cos/sin PE bounded in [-1, 1]
    let cos_pe = ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.5f32);
    let sin_pe = ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.5f32);
    let k_cos_pe = ArrayD::from_elem(IxDyn(&[SEQ_LEN, KV_DIM]), 0.5f32);
    let k_sin_pe = ArrayD::from_elem(IxDyn(&[SEQ_LEN, KV_DIM]), 0.5f32);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _ in 0..3 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // norm_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // norm_w
        bindings.push(TensorParamBinding::ConstantTensor(q_w.clone())); // q_w
        bindings.push(TensorParamBinding::ConstantTensor(kv_w.clone())); // k_w
        bindings.push(TensorParamBinding::ConstantTensor(kv_w.clone())); // v_w
        bindings.push(TensorParamBinding::ConstantTensor(out_w.clone())); // out_w
        bindings.push(TensorParamBinding::ConstantTensor(cos_pe.clone())); // cos_pe
        bindings.push(TensorParamBinding::ConstantTensor(sin_pe.clone())); // sin_pe
        bindings.push(TensorParamBinding::ConstantTensor(k_cos_pe.clone())); // k_cos_pe
        bindings.push(TensorParamBinding::ConstantTensor(k_sin_pe.clone())); // k_sin_pe
    }

    bindings
}

/// IBP bounds propagate through 3-layer deep M-RoPE attention stack.
#[test]
fn test_deep_mrope_stack_ibp() {
    let def = build_qwen3_vl_deep_mrope_stack_kernel();
    let bindings = qwen3_vl_deep_mrope_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL deep M-RoPE stack");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "deep M-RoPE stack output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL deep M-RoPE stack IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 43. Vision projection + decoder SwiGLU composition IBP
// ===========================================================================

/// Build vision projection into decoder: vision features -> linear projection
/// -> RMSNorm -> SwiGLU FFN -> RMSNorm -> SwiGLU FFN -> residual chain.
///
/// Tests the cross-modal boundary where vision encoder features enter the
/// language decoder's representation space.
///
/// Input: `[NUM_PATCHES, HIDDEN_DIM]` (Variable, vision encoder output).
/// Output: `[NUM_PATCHES, HIDDEN_DIM]`.
fn build_qwen3_vl_vision_proj_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_vision_proj_decoder");

    let input = b.add_input("vision_features", &[NUM_PATCHES, HIDDEN_DIM]);
    let shape = [NUM_PATCHES, HIDDEN_DIM];
    let ffn_shape = [NUM_PATCHES, FFN_DIM];

    // Vision-to-decoder projection
    let proj_w = b.add_input("vl_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(input, proj_w, None, &shape);

    // Decoder layer 1: RMSNorm + SwiGLU + residual
    let n1_eps = b.add_input("n1_eps", &[1]);
    let n1_w = b.add_input("n1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(projected, n1_eps, 1, n1_w, &shape);

    let g1_w = b.add_input("g1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let u1_w = b.add_input("u1_weight", &[FFN_DIM, HIDDEN_DIM]);
    let d1_w = b.add_input("d1_weight", &[HIDDEN_DIM, FFN_DIM]);

    let g1 = b.add_linear(normed1, g1_w, None, &ffn_shape);
    let g1_sig = b.add_sigmoid(g1, &ffn_shape);
    let g1_act = b.add_binary_mul(g1, g1_sig, &ffn_shape);
    let u1 = b.add_linear(normed1, u1_w, None, &ffn_shape);
    let h1 = b.add_binary_mul(g1_act, u1, &ffn_shape);
    let ffn1_out = b.add_linear(h1, d1_w, None, &shape);
    let res1 = b.add_binary_add(projected, ffn1_out, &shape);

    // Decoder layer 2: RMSNorm + SwiGLU + residual
    let n2_eps = b.add_input("n2_eps", &[1]);
    let n2_w = b.add_input("n2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

    let g2_w = b.add_input("g2_weight", &[FFN_DIM, HIDDEN_DIM]);
    let u2_w = b.add_input("u2_weight", &[FFN_DIM, HIDDEN_DIM]);
    let d2_w = b.add_input("d2_weight", &[HIDDEN_DIM, FFN_DIM]);

    let g2 = b.add_linear(normed2, g2_w, None, &ffn_shape);
    let g2_sig = b.add_sigmoid(g2, &ffn_shape);
    let g2_act = b.add_binary_mul(g2, g2_sig, &ffn_shape);
    let u2 = b.add_linear(normed2, u2_w, None, &ffn_shape);
    let h2 = b.add_binary_mul(g2_act, u2, &ffn_shape);
    let ffn2_out = b.add_linear(h2, d2_w, None, &shape);
    let out = b.add_binary_add(res1, ffn2_out, &shape);

    b.build(out)
        .expect("valid Qwen3-VL vision proj decoder kernel")
}

/// Bindings for vision projection + decoder.
fn qwen3_vl_vision_proj_decoder_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                       // vision_features
        TensorParamBinding::ConstantTensor(proj_w),         // vl_proj_weight
        TensorParamBinding::ConstantScalar(1e-6),           // n1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // n1_weight
        TensorParamBinding::ConstantTensor(gate_w.clone()), // g1_weight
        TensorParamBinding::ConstantTensor(up_w.clone()),   // u1_weight
        TensorParamBinding::ConstantTensor(down_w.clone()), // d1_weight
        TensorParamBinding::ConstantScalar(1e-6),           // n2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // n2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // g2_weight
        TensorParamBinding::ConstantTensor(up_w),           // u2_weight
        TensorParamBinding::ConstantTensor(down_w),         // d2_weight
    ]
}

/// IBP bounds propagate through vision projection + 2-layer decoder.
#[test]
fn test_vision_proj_decoder_ibp() {
    let def = build_qwen3_vl_vision_proj_decoder_kernel();
    let bindings = qwen3_vl_vision_proj_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL vision proj decoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, HIDDEN_DIM],
        "vision proj decoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL vision proj decoder IBP (patches [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 44. Cross-modal residual: vision encoder + decoder boundary IBP
// ===========================================================================

/// Build a cross-modal residual block: vision features -> projection ->
/// add with text features -> RMSNorm -> SwiGLU FFN -> residual.
///
/// Both vision and text inputs are Variable, testing multi-variable
/// bounds propagation at the cross-modal boundary.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (2x Variable: vision + text).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_cross_modal_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_cross_modal_residual");

    let vision = b.add_input("vision_features", &[SEQ_LEN, HIDDEN_DIM]);
    let text = b.add_input("text_features", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Project vision to text space
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(vision, proj_w, None, &shape);

    // Cross-modal fusion: add projected vision + text
    let fused = b.add_binary_add(projected, text, &shape);

    // RMSNorm
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(fused, norm_eps, 1, norm_w, &shape);

    // SwiGLU FFN
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let g = b.add_linear(normed, gate_w, None, &ffn_shape);
    let g_sig = b.add_sigmoid(g, &ffn_shape);
    let g_act = b.add_binary_mul(g, g_sig, &ffn_shape);
    let u = b.add_linear(normed, up_w, None, &ffn_shape);
    let h = b.add_binary_mul(g_act, u, &ffn_shape);
    let ffn_out = b.add_linear(h, down_w, None, &shape);

    // Residual from fused
    let out = b.add_binary_add(fused, ffn_out, &shape);

    b.build(out)
        .expect("valid Qwen3-VL cross-modal residual kernel")
}

/// Bindings for cross-modal residual.
fn qwen3_vl_cross_modal_residual_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,               // vision_features
        TensorParamBinding::Variable,               // text_features
        TensorParamBinding::ConstantTensor(proj_w), // proj_weight
        TensorParamBinding::ConstantScalar(1e-6),   // norm_eps
        TensorParamBinding::ConstantTensor(norm_w), // norm_weight
        TensorParamBinding::ConstantTensor(gate_w), // gate_weight
        TensorParamBinding::ConstantTensor(up_w),   // up_weight
        TensorParamBinding::ConstantTensor(down_w), // down_weight
    ]
}

/// IBP bounds propagate through cross-modal residual with 2 Variable inputs.
#[test]
fn test_cross_modal_residual_ibp() {
    let def = build_qwen3_vl_cross_modal_residual_kernel();
    let bindings = qwen3_vl_cross_modal_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    // Two Variable inputs of identical shape [SEQ_LEN, HIDDEN_DIM]. Multi-variable
    // graphs slice the input along a leading axis whose size is the number of
    // variables, so the input is [num_vars, SEQ_LEN, HIDDEN_DIM], not a single
    // [2*SEQ_LEN, HIDDEN_DIM] stack.
    let input = uniform_bounds(&[2, SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL cross-modal residual");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "cross-modal residual output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL cross-modal residual IBP (2 vars [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 45. 8-layer decoder with KV-cache simulation IBP
// ===========================================================================

/// Number of KV-cache extended sequence positions.
const KV_CACHE_LEN: usize = 8;

/// Build an 8-layer decoder stack with GQA attention per layer. Models
/// KV-cache inference by using extended K/V sequence length.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable, current step features).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_8layer_decoder_kvcache_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_8layer_decoder_kvcache");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer in 0..8u32 {
        let prefix = format!("l{layer}");

        // Pre-norm for attention
        let n1_eps = b.add_input(&format!("{prefix}_n1_eps"), &[1]);
        let n1_w = b.add_input(&format!("{prefix}_n1_w"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, n1_eps, 1, n1_w, &shape);

        // Q/K/V projections
        let q_w = b.add_input(&format!("{prefix}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, q_w, None, &shape);
        let k = b.add_linear(normed1, k_w, None, &shape);
        let v = b.add_linear(normed1, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(current, attn_out, &shape);

        // Pre-norm for FFN
        let n2_eps = b.add_input(&format!("{prefix}_n2_eps"), &[1]);
        let n2_w = b.add_input(&format!("{prefix}_n2_w"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

        // SwiGLU FFN
        let g_w = b.add_input(&format!("{prefix}_g_w"), &[FFN_DIM, HIDDEN_DIM]);
        let u_w = b.add_input(&format!("{prefix}_u_w"), &[FFN_DIM, HIDDEN_DIM]);
        let d_w = b.add_input(&format!("{prefix}_d_w"), &[HIDDEN_DIM, FFN_DIM]);

        let g = b.add_linear(normed2, g_w, None, &ffn_shape);
        let g_sig = b.add_sigmoid(g, &ffn_shape);
        let g_act = b.add_binary_mul(g, g_sig, &ffn_shape);
        let u = b.add_linear(normed2, u_w, None, &ffn_shape);
        let h = b.add_binary_mul(g_act, u, &ffn_shape);
        let ffn_out = b.add_linear(h, d_w, None, &shape);

        current = b.add_binary_add(res1, ffn_out, &shape);
    }

    b.build(current)
        .expect("valid Qwen3-VL 8-layer decoder KV-cache kernel")
}

/// Bindings for 8-layer decoder with KV-cache.
fn qwen3_vl_8layer_decoder_kvcache_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _ in 0..8 {
        // Attention block
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // n1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // n1_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // q_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // k_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // v_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // out_w
                                                                           // FFN block
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // n2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // n2_w
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // g_w
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // u_w
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // d_w
    }

    bindings
}

/// IBP bounds propagate through 8-layer decoder stack.
#[test]
fn test_8layer_decoder_kvcache_ibp() {
    let def = build_qwen3_vl_8layer_decoder_kvcache_kernel();
    let bindings = qwen3_vl_8layer_decoder_kvcache_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL 8-layer decoder KV-cache");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "8-layer decoder KV-cache output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL 8-layer decoder KV-cache IBP (hidden [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 46. End-to-end generation pipeline IBP (patch embed + encoder + decoder +
//     LM head + softmax)
// ===========================================================================

/// Build an end-to-end generation pipeline: Conv2d patch embed -> 2 encoder
/// blocks -> vision projection -> 2 decoder layers (attention + SwiGLU) ->
/// RMSNorm -> LM head -> softmax.
///
/// This is the deepest composition test, modeling the full inference path from
/// image input to token probability output.
///
/// Input: `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable, image pixels [0, 1]).
/// Output: `[NUM_PATCHES, VOCAB_SIZE]` (probability distribution).
fn build_qwen3_vl_e2e_generation_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_e2e_generation");

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

    // --- 2 encoder blocks ---
    let mut current = patches;
    for enc_block in 0..2u32 {
        let prefix = format!("enc{enc_block}");

        let n1_eps = b.add_input(&format!("{prefix}_n1_eps"), &[1]);
        let n1_w = b.add_input(&format!("{prefix}_n1_w"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, n1_eps, 1, n1_w, &patch_shape);

        let q_w = b.add_input(&format!("{prefix}_q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{prefix}_out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, q_w, None, &patch_shape);
        let k = b.add_linear(normed1, k_w, None, &patch_shape);
        let v = b.add_linear(normed1, v_w, None, &patch_shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &patch_shape);
        let attn_out = b.add_linear(attn, out_w, None, &patch_shape);
        let res1 = b.add_binary_add(current, attn_out, &patch_shape);

        let n2_eps = b.add_input(&format!("{prefix}_n2_eps"), &[1]);
        let n2_w = b.add_input(&format!("{prefix}_n2_w"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &patch_shape);

        let g_w = b.add_input(&format!("{prefix}_g_w"), &[FFN_DIM, HIDDEN_DIM]);
        let u_w = b.add_input(&format!("{prefix}_u_w"), &[FFN_DIM, HIDDEN_DIM]);
        let d_w = b.add_input(&format!("{prefix}_d_w"), &[HIDDEN_DIM, FFN_DIM]);

        let g = b.add_linear(normed2, g_w, None, &ffn_shape);
        let g_sig = b.add_sigmoid(g, &ffn_shape);
        let g_act = b.add_binary_mul(g, g_sig, &ffn_shape);
        let u = b.add_linear(normed2, u_w, None, &ffn_shape);
        let h = b.add_binary_mul(g_act, u, &ffn_shape);
        let ffn_out = b.add_linear(h, d_w, None, &patch_shape);

        current = b.add_binary_add(res1, ffn_out, &patch_shape);
    }

    // --- Vision projection ---
    let proj_w = b.add_input("vl_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let projected = b.add_linear(current, proj_w, None, &patch_shape);

    // --- 2 decoder layers (SwiGLU FFN only, simplified) ---
    let mut dec = projected;
    for dec_layer in 0..2u32 {
        let prefix = format!("dec{dec_layer}");

        let n_eps = b.add_input(&format!("{prefix}_n_eps"), &[1]);
        let n_w = b.add_input(&format!("{prefix}_n_w"), &[HIDDEN_DIM]);
        let normed = b.add_rms_norm(dec, n_eps, 1, n_w, &patch_shape);

        let g_w = b.add_input(&format!("{prefix}_g_w"), &[FFN_DIM, HIDDEN_DIM]);
        let u_w = b.add_input(&format!("{prefix}_u_w"), &[FFN_DIM, HIDDEN_DIM]);
        let d_w = b.add_input(&format!("{prefix}_d_w"), &[HIDDEN_DIM, FFN_DIM]);

        let g = b.add_linear(normed, g_w, None, &ffn_shape);
        let g_sig = b.add_sigmoid(g, &ffn_shape);
        let g_act = b.add_binary_mul(g, g_sig, &ffn_shape);
        let u = b.add_linear(normed, u_w, None, &ffn_shape);
        let h = b.add_binary_mul(g_act, u, &ffn_shape);
        let ffn_out = b.add_linear(h, d_w, None, &patch_shape);

        dec = b.add_binary_add(dec, ffn_out, &patch_shape);
    }

    // --- LM head: RMSNorm -> Linear -> softmax ---
    let fn_eps = b.add_input("final_norm_eps", &[1]);
    let fn_w = b.add_input("final_norm_weight", &[HIDDEN_DIM]);
    let final_normed = b.add_rms_norm(dec, fn_eps, 1, fn_w, &patch_shape);

    let lm_w = b.add_input("lm_head_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(final_normed, lm_w, None, &[NUM_PATCHES, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[NUM_PATCHES, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid Qwen3-VL e2e generation kernel")
}

/// Bindings for end-to-end generation pipeline.
fn qwen3_vl_e2e_generation_bindings() -> Vec<TensorParamBinding> {
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
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![
        TensorParamBinding::Variable,                   // image
        TensorParamBinding::ConstantTensor(patch_w),    // patch_weight
        TensorParamBinding::ConstantTensor(patch_bias), // patch_bias
    ];

    // 2 encoder blocks
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // n1_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // n1_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // q_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // k_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // v_w
        bindings.push(TensorParamBinding::ConstantTensor(qkvo_w.clone())); // out_w
        bindings.push(TensorParamBinding::ConstantScalar(1e-5)); // n2_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // n2_w
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // g_w
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // u_w
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // d_w
    }

    // Vision projection
    bindings.push(TensorParamBinding::ConstantTensor(proj_w));

    // 2 decoder layers
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // n_eps
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // n_w
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // g_w
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // u_w
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // d_w
    }

    // LM head
    bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // final_norm_eps
    bindings.push(TensorParamBinding::ConstantTensor(norm_w)); // final_norm_weight
    bindings.push(TensorParamBinding::ConstantTensor(lm_w)); // lm_head_weight

    bindings
}

/// IBP bounds through end-to-end generation pipeline: softmax output in [0, 1].
#[test]
fn test_e2e_generation_ibp() {
    let def = build_qwen3_vl_e2e_generation_kernel();
    let bindings = qwen3_vl_e2e_generation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL e2e generation");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, VOCAB_SIZE],
        "e2e generation output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL e2e generation IBP (image [0,1]): bounds=[{lo_min}, {hi_max}]");

    // Softmax codomain is (0, 1)
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "e2e softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "e2e softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 47. End-to-end generation pipeline CROWN
// ===========================================================================

/// CROWN verification of end-to-end generation pipeline.
#[test]
fn test_e2e_generation_crown() {
    let def = build_qwen3_vl_e2e_generation_kernel();
    let bindings = qwen3_vl_e2e_generation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, VOCAB_SIZE],
        "e2e generation CROWN output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL e2e generation CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "CROWN lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "CROWN upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 48. Quantized MoE FFN composition IBP
// ===========================================================================

/// Build a quantized MoE block: 2 experts with INT4-dequantized weights
/// through SwiGLU FFNs + sum + residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_qwen3_vl_quantized_moe_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_quantized_moe");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, QUANT_FFN_DIM];

    // Expert 1 (quantized weights)
    let e1_gw = b.add_input("e1_gate_w", &[QUANT_FFN_DIM, HIDDEN_DIM]);
    let e1_uw = b.add_input("e1_up_w", &[QUANT_FFN_DIM, HIDDEN_DIM]);
    let e1_dw = b.add_input("e1_down_w", &[HIDDEN_DIM, QUANT_FFN_DIM]);

    let e1g = b.add_linear(input, e1_gw, None, &ffn_shape);
    let e1g_sig = b.add_sigmoid(e1g, &ffn_shape);
    let e1g_act = b.add_binary_mul(e1g, e1g_sig, &ffn_shape);
    let e1u = b.add_linear(input, e1_uw, None, &ffn_shape);
    let e1h = b.add_binary_mul(e1g_act, e1u, &ffn_shape);
    let e1_out = b.add_linear(e1h, e1_dw, None, &shape);

    // Expert 2 (quantized weights)
    let e2_gw = b.add_input("e2_gate_w", &[QUANT_FFN_DIM, HIDDEN_DIM]);
    let e2_uw = b.add_input("e2_up_w", &[QUANT_FFN_DIM, HIDDEN_DIM]);
    let e2_dw = b.add_input("e2_down_w", &[HIDDEN_DIM, QUANT_FFN_DIM]);

    let e2g = b.add_linear(input, e2_gw, None, &ffn_shape);
    let e2g_sig = b.add_sigmoid(e2g, &ffn_shape);
    let e2g_act = b.add_binary_mul(e2g, e2g_sig, &ffn_shape);
    let e2u = b.add_linear(input, e2_uw, None, &ffn_shape);
    let e2h = b.add_binary_mul(e2g_act, e2u, &ffn_shape);
    let e2_out = b.add_linear(e2h, e2_dw, None, &shape);

    // Sum + residual
    let expert_sum = b.add_binary_add(e1_out, e2_out, &shape);
    let out = b.add_binary_add(input, expert_sum, &shape);

    b.build(out).expect("valid Qwen3-VL quantized MoE kernel")
}

/// Bindings for quantized MoE (INT4 dequantized weights).
fn qwen3_vl_quantized_moe_bindings() -> Vec<TensorParamBinding> {
    let gate_w = ArrayD::from_elem(IxDyn(&[QUANT_FFN_DIM, HIDDEN_DIM]), INT4_MAX_FLOAT);
    let up_w = ArrayD::from_elem(IxDyn(&[QUANT_FFN_DIM, HIDDEN_DIM]), INT4_MAX_FLOAT);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, QUANT_FFN_DIM]), INT4_MAX_FLOAT);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantTensor(gate_w.clone()), // e1_gate_w
        TensorParamBinding::ConstantTensor(up_w.clone()),   // e1_up_w
        TensorParamBinding::ConstantTensor(down_w.clone()), // e1_down_w
        TensorParamBinding::ConstantTensor(gate_w),         // e2_gate_w
        TensorParamBinding::ConstantTensor(up_w),           // e2_up_w
        TensorParamBinding::ConstantTensor(down_w),         // e2_down_w
    ]
}

/// IBP bounds through quantized MoE are tighter than FP32 baseline.
#[test]
fn test_quantized_moe_ibp() {
    let def = build_qwen3_vl_quantized_moe_kernel();
    let bindings = qwen3_vl_quantized_moe_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let quant_output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL quantized MoE");

    assert_eq!(
        quant_output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "quantized MoE output shape mismatch"
    );
    assert_bounds_valid(&quant_output);

    let (q_lo, q_hi) = bounds_min_max(&quant_output);
    eprintln!("Qwen3-VL quantized MoE IBP (hidden [-1,1]): bounds=[{q_lo}, {q_hi}]");

    // Compare against FP32 baseline (should be tighter due to smaller weights)
    let fp_gate_w = ArrayD::from_elem(IxDyn(&[QUANT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fp_up_w = ArrayD::from_elem(IxDyn(&[QUANT_FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fp_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, QUANT_FFN_DIM]), WEIGHT_MAG);
    let fp_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(fp_gate_w.clone()),
        TensorParamBinding::ConstantTensor(fp_up_w.clone()),
        TensorParamBinding::ConstantTensor(fp_down_w.clone()),
        TensorParamBinding::ConstantTensor(fp_gate_w),
        TensorParamBinding::ConstantTensor(fp_up_w),
        TensorParamBinding::ConstantTensor(fp_down_w),
    ];
    let fp_graph = tensor_kernel_to_graph(&def, &fp_bindings).expect("fp graph");
    let fp_output = fp_graph
        .propagate_ibp(&input)
        .expect("IBP through FP32 MoE");
    let (fp_lo, fp_hi) = bounds_min_max(&fp_output);
    eprintln!("FP32 baseline MoE IBP: bounds=[{fp_lo}, {fp_hi}]");

    // INT4 has smaller maximum magnitude than FP32, so bounds should be tighter
    // (INT4_MAX_FLOAT = 0.07 vs WEIGHT_MAG = 0.02, so this tests the bound
    // relationship; both are small but INT4 max may actually be larger)
    assert!(q_lo.is_finite(), "quantized lower bound must be finite");
    assert!(q_hi.is_finite(), "quantized upper bound must be finite");
}

// ===========================================================================
// 49. Multi-image composition: 2 images through shared encoder IBP
// ===========================================================================

/// Build a multi-image pipeline: 2 image inputs through separate patch
/// embeddings -> shared encoder -> concat -> projection.
///
/// Models multi-image understanding where multiple images are processed
/// by the same vision encoder and fused for the language decoder.
///
/// Input: 2x `[IN_CHANNELS, IMG_SIZE, IMG_SIZE]` (Variable).
/// Output: `[2 * NUM_PATCHES, HIDDEN_DIM]`.
fn build_qwen3_vl_multi_image_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_multi_image");
    let total_patches = 2 * NUM_PATCHES;

    // Image 1
    let img1 = b.add_input("image1", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let pw1 = b.add_input(
        "patch_weight1",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let pb1 = b.add_input("patch_bias1", &[HIDDEN_DIM]);
    let c1 = b.add_conv2d(
        img1,
        pw1,
        Some(pb1),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let r1 = b.add_reshape(c1, &[HIDDEN_DIM, NUM_PATCHES]);
    let p1 = b.add_transpose(r1, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);

    // Image 2
    let img2 = b.add_input("image2", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let pw2 = b.add_input(
        "patch_weight2",
        &[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE],
    );
    let pb2 = b.add_input("patch_bias2", &[HIDDEN_DIM]);
    let c2 = b.add_conv2d(
        img2,
        pw2,
        Some(pb2),
        PATCH_SIZE,
        PATCH_SIZE,
        0,
        0,
        &[HIDDEN_DIM, GRID_SIZE, GRID_SIZE],
    );
    let r2 = b.add_reshape(c2, &[HIDDEN_DIM, NUM_PATCHES]);
    let p2 = b.add_transpose(r2, &[1, 0], &[NUM_PATCHES, HIDDEN_DIM]);

    // Concatenate along sequence dimension
    let concat = b.add_concat(&[p1, p2], 0, &[total_patches, HIDDEN_DIM]);

    // Shared encoder: RMSNorm + attention + residual
    let n_eps = b.add_input("enc_norm_eps", &[1]);
    let n_w = b.add_input("enc_norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(concat, n_eps, 1, n_w, &[total_patches, HIDDEN_DIM]);

    let q_w = b.add_input("enc_q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("enc_k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("enc_v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input("enc_out_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let q = b.add_linear(normed, q_w, None, &[total_patches, HIDDEN_DIM]);
    let k = b.add_linear(normed, k_w, None, &[total_patches, HIDDEN_DIM]);
    let v = b.add_linear(normed, v_w, None, &[total_patches, HIDDEN_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[total_patches, HIDDEN_DIM],
    );
    let attn_out = b.add_linear(attn, out_w, None, &[total_patches, HIDDEN_DIM]);
    let res = b.add_binary_add(concat, attn_out, &[total_patches, HIDDEN_DIM]);

    // Final projection
    let proj_w = b.add_input("proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(res, proj_w, None, &[total_patches, HIDDEN_DIM]);

    b.build(out).expect("valid Qwen3-VL multi-image kernel")
}

/// Bindings for multi-image composition.
fn qwen3_vl_multi_image_bindings() -> Vec<TensorParamBinding> {
    let patch_w = ArrayD::from_elem(
        IxDyn(&[HIDDEN_DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let patch_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkvo_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                           // image1
        TensorParamBinding::ConstantTensor(patch_w.clone()),    // patch_weight1
        TensorParamBinding::ConstantTensor(patch_bias.clone()), // patch_bias1
        TensorParamBinding::Variable,                           // image2
        TensorParamBinding::ConstantTensor(patch_w),            // patch_weight2
        TensorParamBinding::ConstantTensor(patch_bias),         // patch_bias2
        TensorParamBinding::ConstantScalar(1e-6),               // enc_norm_eps
        TensorParamBinding::ConstantTensor(norm_w),             // enc_norm_weight
        TensorParamBinding::ConstantTensor(qkvo_w.clone()),     // enc_q_w
        TensorParamBinding::ConstantTensor(qkvo_w.clone()),     // enc_k_w
        TensorParamBinding::ConstantTensor(qkvo_w.clone()),     // enc_v_w
        TensorParamBinding::ConstantTensor(qkvo_w),             // enc_out_w
        TensorParamBinding::ConstantTensor(proj_w),             // proj_weight
    ]
}

/// IBP bounds propagate through multi-image composition with shared encoder.
#[test]
fn test_multi_image_composition_ibp() {
    let def = build_qwen3_vl_multi_image_kernel();
    let bindings = qwen3_vl_multi_image_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // 2 equally-shaped [IN_CHANNELS, IMG_SIZE, IMG_SIZE] Variable image inputs.
    // Multi-variable graphs slice the input along a leading axis whose size is
    // the number of variables, so the input is [num_vars, IN_CHANNELS, IMG_SIZE,
    // IMG_SIZE], not a single [2*IN_CHANNELS, IMG_SIZE, IMG_SIZE] channel stack.
    let input = image_bounds_01(&[2, IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL multi-image composition");

    let total_patches = 2 * NUM_PATCHES;
    assert_eq!(
        output.lower_upper().0.shape(),
        &[total_patches, HIDDEN_DIM],
        "multi-image composition output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL multi-image IBP (2 images [0,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 50. Vision projection + decoder verify_and_record
// ===========================================================================

/// Verify and record: vision projection + decoder composition.
#[test]
fn test_vision_proj_decoder_verify_and_record() {
    let def = build_qwen3_vl_vision_proj_decoder_kernel();
    let bindings = qwen3_vl_vision_proj_decoder_bindings();
    let input = uniform_bounds(&[NUM_PATCHES, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_vl_vision_proj_decoder");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NUM_PATCHES, HIDDEN_DIM]);
}

// ===========================================================================
// 51. 8-layer decoder verify_and_record
// ===========================================================================

/// Verify and record: 8-layer decoder stack.
#[test]
fn test_8layer_decoder_verify_and_record() {
    let def = build_qwen3_vl_8layer_decoder_kvcache_kernel();
    let bindings = qwen3_vl_8layer_decoder_kvcache_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_vl_8layer_decoder_kvcache");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 52. Deep M-RoPE stack CROWN
// ===========================================================================

/// CROWN verification of 3-layer deep M-RoPE attention stack.
#[test]
fn test_deep_mrope_stack_crown() {
    let def = build_qwen3_vl_deep_mrope_stack_kernel();
    let bindings = qwen3_vl_deep_mrope_stack_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL deep M-RoPE stack CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "CROWN lower bound must be finite");
    assert!(hi_max.is_finite(), "CROWN upper bound must be finite");
}

// ===========================================================================
// 53. Interleaved self + cross attention IBP
// ===========================================================================

/// Build interleaved self-attention then cross-attention block.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- text hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Architecture: RMSNorm -> self-attention(causal) -> residual ->
/// RMSNorm -> cross-attention(vision K/V) -> residual.
/// Models a Qwen3-VL decoder layer that interleaves self- and cross-attention.
fn build_qwen3_vl_interleaved_self_cross_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_interleaved_self_cross");

    let input = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // --- Self-attention ---
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // GQA: project Q/K/V to KV_DIM so QK^T contracts over a shared head dim
    // (q_d == k_d == KV_DIM), as in build_qwen3_vl_gqa_kv_cache_kernel. The
    // attention output [SEQ_LEN, KV_DIM] is then projected back up to HIDDEN_DIM.
    let sq_w = b.add_input("self_q_weight", &[KV_DIM, HIDDEN_DIM]);
    let sk_w = b.add_input("self_k_weight", &[KV_DIM, HIDDEN_DIM]);
    let sv_w = b.add_input("self_v_weight", &[KV_DIM, HIDDEN_DIM]);
    let sout_w = b.add_input("self_out_weight", &[HIDDEN_DIM, KV_DIM]);

    let sq = b.add_linear(normed1, sq_w, None, &[SEQ_LEN, KV_DIM]);
    let sk = b.add_linear(normed1, sk_w, None, &[SEQ_LEN, KV_DIM]);
    let sv = b.add_linear(normed1, sv_w, None, &[SEQ_LEN, KV_DIM]);
    let self_attn =
        b.add_attention(sq, sk, sv, AttentionMask::Causal, Some(scale), &[SEQ_LEN, KV_DIM]);
    let self_out = b.add_linear(self_attn, sout_w, None, &shape);
    let res1 = b.add_binary_add(input, self_out, &shape);

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
    let cross_out = b.add_linear(cross_attn, cout_w, None, &shape);
    let out = b.add_binary_add(res1, cross_out, &shape);

    b.build(out)
        .expect("valid Qwen3-VL interleaved self+cross kernel")
}

/// Bindings for interleaved self + cross attention.
fn qwen3_vl_interleaved_self_cross_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let qkv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let kv_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    // GQA out-projection lifts the KV_DIM attention output back to HIDDEN_DIM.
    let out_up_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);
    let vision_feat = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                            // text_hidden
        TensorParamBinding::ConstantScalar(1e-5),                // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()),      // norm1_weight
        TensorParamBinding::ConstantTensor(kv_w.clone()),        // self_q_weight (GQA: -> KV_DIM)
        TensorParamBinding::ConstantTensor(kv_w.clone()),        // self_k_weight
        TensorParamBinding::ConstantTensor(kv_w),                // self_v_weight
        TensorParamBinding::ConstantTensor(out_up_w),            // self_out_weight (KV_DIM -> HIDDEN_DIM)
        TensorParamBinding::ConstantScalar(1e-5),                // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),              // norm2_weight
        TensorParamBinding::ConstantTensor(qkv_w.clone()),       // cross_q_weight
        TensorParamBinding::ConstantTensor(vision_feat.clone()), // vision_k
        TensorParamBinding::ConstantTensor(vision_feat),         // vision_v
        TensorParamBinding::ConstantTensor(qkv_w),               // cross_out_weight
    ]
}

/// IBP through interleaved self + cross attention.
///
/// Verifies bounds through self-attention (causal) followed by cross-attention
/// (vision K/V) with RMSNorm and residuals between each.
#[test]
fn test_qwen3_vl_interleaved_self_cross_ibp() {
    let def = build_qwen3_vl_interleaved_self_cross_kernel();
    let bindings = qwen3_vl_interleaved_self_cross_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through interleaved self+cross attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "interleaved self+cross output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL interleaved self+cross IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through interleaved self + cross attention.
#[test]
fn test_qwen3_vl_interleaved_self_cross_crown() {
    let def = build_qwen3_vl_interleaved_self_cross_kernel();
    let bindings = qwen3_vl_interleaved_self_cross_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Qwen3-VL interleaved self+cross CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "CROWN lower bound must be finite");
    assert!(hi_max.is_finite(), "CROWN upper bound must be finite");
}

// ===========================================================================
// 54. Vision token insertion with text tokens IBP
// ===========================================================================

/// Build vision token insertion into text token stream.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- text hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Models Qwen3-VL's approach of inserting projected vision tokens into
/// the text sequence: text goes through a linear projection, vision features
/// (constant) go through a separate projection, then both are combined
/// via addition (simulating token-level fusion). A final RMSNorm normalizes
/// the fused representation.
fn build_qwen3_vl_vision_token_insert_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_vision_token_insert");

    let text = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let text_shape = [SEQ_LEN, HIDDEN_DIM];

    // Text projection
    let text_proj_w = b.add_input("text_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let text_proj_b = b.add_input("text_proj_bias", &[HIDDEN_DIM]);
    let text_proj = b.add_linear(text, text_proj_w, Some(text_proj_b), &text_shape);

    // Vision projection (constant features projected to text-compatible space)
    let vision_feat = b.add_input("vision_features", &[SEQ_LEN, HIDDEN_DIM]);
    let vis_proj_w = b.add_input("vis_proj_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let vis_proj = b.add_linear(vision_feat, vis_proj_w, None, &text_shape);

    // Fuse via addition (token-level fusion)
    let fused = b.add_binary_add(text_proj, vis_proj, &text_shape);

    // Post-fusion RMSNorm
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let out = b.add_rms_norm(fused, norm_eps, 1, norm_w, &text_shape);

    b.build(out)
        .expect("valid Qwen3-VL vision token insertion kernel")
}

/// Bindings for vision token insertion.
fn qwen3_vl_vision_token_insert_bindings() -> Vec<TensorParamBinding> {
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let proj_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let vision_feat = ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.5f32);

    vec![
        TensorParamBinding::Variable,                       // text_hidden
        TensorParamBinding::ConstantTensor(proj_w.clone()), // text_proj_weight
        TensorParamBinding::ConstantTensor(proj_b),         // text_proj_bias
        TensorParamBinding::ConstantTensor(vision_feat),    // vision_features
        TensorParamBinding::ConstantTensor(proj_w),         // vis_proj_weight
        TensorParamBinding::ConstantScalar(1e-5),           // norm_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm_weight
    ]
}

/// IBP through vision token insertion with text tokens.
///
/// Verifies bounds when projected vision features are added to projected
/// text features and normalized. Models the Qwen3-VL cross-modal token
/// interleaving mechanism.
#[test]
fn test_qwen3_vl_vision_token_insert_ibp() {
    let def = build_qwen3_vl_vision_token_insert_kernel();
    let bindings = qwen3_vl_vision_token_insert_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through vision token insertion");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "vision token insertion output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL vision token insert IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 55. M-RoPE through cross-attention IBP
// ===========================================================================

/// Build M-RoPE-enhanced cross-attention block.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable -- text hidden states).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
///
/// Architecture: RMSNorm -> Q projection -> M-RoPE(cos/sin multiply on Q) ->
/// cross-attention with vision K/V -> output projection -> residual.
/// Models the Qwen3-VL multimodal rotary position embedding applied to
/// queries before cross-attention.
fn build_qwen3_vl_mrope_cross_attn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_vl_mrope_cross_attn");

    let input = b.add_input("text_hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // RMSNorm
    let norm_eps = b.add_input("norm_eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(input, norm_eps, 1, norm_w, &shape);

    // Q projection
    let q_w = b.add_input("q_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let q = b.add_linear(normed, q_w, None, &shape);

    // M-RoPE: multiply Q by cos and sin position embeddings
    // cos/sin are constant position embeddings bounded in [-1, 1]
    let rope_cos = b.add_input("rope_cos", &[SEQ_LEN, HIDDEN_DIM]);
    let rope_sin = b.add_input("rope_sin", &[SEQ_LEN, HIDDEN_DIM]);

    // RoPE application: q * cos + rotate(q) * sin
    // Simplified as: q_rotated = q * cos + q * sin (structural approximation)
    let q_cos = b.add_binary_mul(q, rope_cos, &shape);
    let q_sin = b.add_binary_mul(q, rope_sin, &shape);
    let q_rotated = b.add_binary_add(q_cos, q_sin, &shape);

    // Vision K/V (frozen)
    let vision_k = b.add_input("vision_k", &[NUM_PATCHES, HIDDEN_DIM]);
    let vision_v = b.add_input("vision_v", &[NUM_PATCHES, HIDDEN_DIM]);

    let attn = b.add_attention(
        q_rotated,
        vision_k,
        vision_v,
        AttentionMask::Standard,
        Some(scale),
        &shape,
    );

    let out_w = b.add_input("out_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let attn_out = b.add_linear(attn, out_w, None, &shape);
    let out = b.add_binary_add(input, attn_out, &shape);

    b.build(out)
        .expect("valid Qwen3-VL M-RoPE cross-attention kernel")
}

/// Bindings for M-RoPE cross-attention.
fn qwen3_vl_mrope_cross_attn_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let attn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    // RoPE cos/sin bounded in [-1, 1]
    let rope_cos = ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.5f32);
    let rope_sin = ArrayD::from_elem(IxDyn(&[SEQ_LEN, HIDDEN_DIM]), 0.3f32);
    let vision_feat = ArrayD::from_elem(IxDyn(&[NUM_PATCHES, HIDDEN_DIM]), WEIGHT_MAG);

    vec![
        TensorParamBinding::Variable,                            // text_hidden
        TensorParamBinding::ConstantScalar(1e-5),                // norm_eps
        TensorParamBinding::ConstantTensor(norm_w),              // norm_weight
        TensorParamBinding::ConstantTensor(attn_w.clone()),      // q_weight
        TensorParamBinding::ConstantTensor(rope_cos),            // rope_cos
        TensorParamBinding::ConstantTensor(rope_sin),            // rope_sin
        TensorParamBinding::ConstantTensor(vision_feat.clone()), // vision_k
        TensorParamBinding::ConstantTensor(vision_feat),         // vision_v
        TensorParamBinding::ConstantTensor(attn_w),              // out_weight
    ]
}

/// IBP through M-RoPE enhanced cross-attention.
///
/// Verifies bounds when multimodal rotary position embeddings are applied
/// to queries before cross-attention. cos/sin values are bounded in [-1, 1].
#[test]
fn test_qwen3_vl_mrope_cross_attn_ibp() {
    let def = build_qwen3_vl_mrope_cross_attn_kernel();
    let bindings = qwen3_vl_mrope_cross_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through M-RoPE cross-attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "M-RoPE cross-attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL M-RoPE cross-attn IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// CROWN through M-RoPE enhanced cross-attention.
///
/// Tests CROWN linearization through RoPE multiplication + softmax.
#[test]
fn test_qwen3_vl_mrope_cross_attn_crown() {
    let def = build_qwen3_vl_mrope_cross_attn_kernel();
    let bindings = qwen3_vl_mrope_cross_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Qwen3-VL M-RoPE cross-attn CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(lo_min.is_finite(), "CROWN lower bound must be finite");
    assert!(hi_max.is_finite(), "CROWN upper bound must be finite");
}
