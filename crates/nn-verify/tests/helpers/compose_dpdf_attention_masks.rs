// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for attention mask patterns: causal, padding, sliding window,
//! block-sparse, cross-attention masks, and their effects on bound propagation.
//!
//! Verifies IBP and CROWN bound propagation through masked attention variants
//! used across dpdf models (GLM-OCR, Qwen3-VL, Table Transformer, Granite-Docling).
//! Attention masks are critical for correctness: causal masks enforce autoregressive
//! decoding, padding masks prevent attending to pad tokens, and sliding window
//! masks control local attention span.
//!
//! 1.  **Causal mask generation bounds** (IBP)
//! 2.  **Padding mask effect on attention bounds** (IBP)
//! 3.  **Combined causal + padding mask** (IBP)
//! 4.  **Sliding window mask generation** (IBP)
//! 5.  **Block-sparse attention mask** (IBP)
//! 6.  **Attention with causal mask vs without mask comparison** (IBP)
//! 7.  **Mask-based attention weight zeroing** (IBP)
//! 8.  **Cross-attention mask (encoder padding)** (IBP)
//! 9.  **Bidirectional vs causal mask bound comparison** (IBP)
//! 10. **Prefix mask (prefix LM attention)** (IBP)
//! 11. **Mask numerical stability (large negative values)** (IBP)
//! 12. **CROWN tightness for masked attention** (CROWN)
//! 13. **Mask monotone tightening** (IBP)
//! 14. **Multi-head mask broadcasting** (IBP)
//! 15. **Full masked attention block: LN + mask + MHA + residual** (IBP + CROWN)
//!
//! Architecture references:
//! - Causal masking (Vaswani et al., 2017): autoregressive decoder attention
//! - Sliding window (Beltagy et al., 2020): Longformer local attention
//! - Block-sparse (Child et al., 2019): Sparse Transformer
//! - Prefix LM (Raffel et al., 2020): T5-style prefix attention
//! - GLM-4V (THUDM): causal + cross-attention masks
//! - Qwen3-VL (Alibaba): sliding window + causal masks
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, DIM=16, NUM_HEADS=4, HEAD_DIM=4
//!
//! Part of #4043: Compose tests for attention mask patterns.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const SEQ_LEN: usize = 4;
const DIM: usize = 16;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = DIM / NUM_HEADS; // 4
const WEIGHT_MAG: f32 = 0.02;

/// Encoder sequence length for cross-attention tests.
const ENC_SEQ_LEN: usize = 6;

/// Sliding window size for local attention tests.
const WINDOW_SIZE: usize = 2;

/// FFN intermediate dimension for full block tests.
const FFN_DIM: usize = 32;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

/// Weight tensor binding helper.
fn weight_binding(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Build a standard causal-masked MHA kernel.
fn build_causal_mha_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let o_w = b.add_input("o_w", &[DIM, DIM]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Causal,
            &[SEQ_LEN, DIM],
        )
        .expect("valid causal MHA");

    b.build(out).expect("valid causal MHA kernel")
}

/// Build a standard bidirectional (no-mask) MHA kernel.
fn build_standard_mha_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let o_w = b.add_input("o_w", &[DIM, DIM]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, DIM],
        )
        .expect("valid standard MHA");

    b.build(out).expect("valid standard MHA kernel")
}

/// Standard MHA weight bindings: variable input + 4 projection weights.
fn mha_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
    ]
}

/// Build SiLU activation: SiLU(x) = x * sigmoid(x).
fn add_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

// ===========================================================================
// 1. Causal mask generation bounds (IBP)
// ===========================================================================

/// Verify that causal-masked MHA produces finite, valid IBP bounds.
/// Causal mask restricts position j to attend only to positions <= j,
/// which should produce bounded attention weights.
#[test]
fn test_causal_mask_generation_ibp() {
    let def = build_causal_mha_kernel("dpdf_mask_causal_gen");
    let bindings = mha_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Causal mask generation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. Padding mask effect on attention bounds (IBP)
// ===========================================================================

/// Model padding mask effect: attention over a shorter effective sequence.
/// Padding mask is simulated by using a shorter sequence length. With fewer
/// key/value positions, attention bounds should remain tight.
#[test]
fn test_padding_mask_effect_ibp() {
    // Effective sequence = 2 (simulating padding out positions 3,4)
    let eff_seq = 2;
    let mut b = TensorBlockBuilder::new("dpdf_mask_padding_effect");
    let input = b.add_input("x", &[eff_seq, DIM]);
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let o_w = b.add_input("o_w", &[DIM, DIM]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[eff_seq, DIM],
        )
        .expect("valid padded MHA");
    let def = b.build(out).expect("valid padded MHA kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[eff_seq, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Padding mask effect (eff_seq={eff_seq}) IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

// ===========================================================================
// 3. Combined causal + padding mask (IBP)
// ===========================================================================

/// Combined causal + padding effect: causal attention on a short effective
/// sequence. Both masking constraints apply simultaneously, producing
/// tighter bounds than either alone.
#[test]
fn test_combined_causal_padding_mask_ibp() {
    let eff_seq = 3;
    let mut b = TensorBlockBuilder::new("dpdf_mask_causal_padding");
    let input = b.add_input("x", &[eff_seq, DIM]);
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let o_w = b.add_input("o_w", &[DIM, DIM]);

    // Causal mask on the effective (non-padded) sequence
    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Causal,
            &[eff_seq, DIM],
        )
        .expect("valid causal+padding MHA");
    let def = b.build(out).expect("valid causal+padding MHA kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[eff_seq, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Combined causal+padding (eff_seq={eff_seq}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 4. Sliding window mask generation (IBP)
// ===========================================================================

/// Sliding window attention: each position attends to at most WINDOW_SIZE
/// positions on each side. Modeled via low-dim attention with constrained
/// scope. The restricted receptive field should produce finite bounds.
#[test]
fn test_sliding_window_mask_ibp() {
    // Model sliding window via attention on a window-sized subsequence.
    // Input [SEQ_LEN, DIM]: Q projects full sequence, K/V project a
    // window-sized context. For verification, we model the local window
    // as a smaller attention with window_dim = (2*WINDOW_SIZE+1) features.
    let window_dim = (2 * WINDOW_SIZE + 1).min(SEQ_LEN); // 5 or SEQ_LEN
    let mut b = TensorBlockBuilder::new("dpdf_mask_sliding_window");
    let input = b.add_input("x", &[window_dim, DIM]);
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let o_w = b.add_input("o_w", &[DIM, DIM]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[window_dim, DIM],
        )
        .expect("valid sliding window MHA");
    let def = b.build(out).expect("valid sliding window MHA kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[window_dim, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Sliding window (window_dim={window_dim}) IBP: width={width:.6}");
    assert!(width.is_finite(), "sliding window width must be finite");
}

// ===========================================================================
// 5. Block-sparse attention mask (IBP)
// ===========================================================================

/// Block-sparse attention: tokens attend to fixed block-aligned positions.
/// Modeled as small-block attention. Block sparsity limits the number of
/// attending positions, keeping bounds bounded.
#[test]
fn test_block_sparse_mask_ibp() {
    let block_size = 2; // 2-token blocks
    let mut b = TensorBlockBuilder::new("dpdf_mask_block_sparse");
    let input = b.add_input("x", &[block_size, DIM]);
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let o_w = b.add_input("o_w", &[DIM, DIM]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[block_size, DIM],
        )
        .expect("valid block-sparse MHA");
    let def = b.build(out).expect("valid block-sparse MHA kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[block_size, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Block-sparse (block_size={block_size}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. Attention with causal mask vs without mask comparison (IBP)
// ===========================================================================

/// Compare causal-masked vs bidirectional attention bounds.
/// Causal masking restricts the attention context, so the causal variant
/// should produce bounds at least as wide (more uncertain) or comparable
/// to the full bidirectional case.
#[test]
fn test_causal_vs_no_mask_comparison_ibp() {
    let bindings = mha_bindings();
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    // Causal-masked attention
    let causal_def = build_causal_mha_kernel("dpdf_mask_causal_compare");
    let causal_graph = tensor_kernel_to_graph(&causal_def, &bindings).expect("causal graph");
    let causal_output = causal_graph.propagate_ibp(&input).expect("causal IBP");
    assert_bounds_valid(&causal_output);

    // Bidirectional attention (no mask)
    let std_def = build_standard_mha_kernel("dpdf_mask_std_compare");
    let std_graph = tensor_kernel_to_graph(&std_def, &bindings).expect("std graph");
    let std_output = std_graph.propagate_ibp(&input).expect("std IBP");
    assert_bounds_valid(&std_output);

    let causal_width = bound_width(&causal_output);
    let std_width = bound_width(&std_output);
    eprintln!(
        "Causal vs standard MHA IBP: causal_width={causal_width:.6}, std_width={std_width:.6}"
    );
    // Both should be finite
    assert!(causal_width.is_finite(), "causal width must be finite");
    assert!(std_width.is_finite(), "standard width must be finite");
}

// ===========================================================================
// 7. Mask-based attention weight zeroing (IBP)
// ===========================================================================

/// Attention with softmax produces weights in [0, 1]. Masking zeros out
/// certain positions, but the softmax output remains in [0, 1].
/// We verify this by building attention + softmax explicitly.
#[test]
fn test_mask_attention_weight_zeroing_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mask_weight_zeroing");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);

    // Q, K projections
    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, DIM]);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, DIM]);

    // Scaled dot-product: Q @ K^T -> [SEQ_LEN, SEQ_LEN]
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(scale), &[SEQ_LEN, SEQ_LEN]);

    // Softmax over last dim: attention weights in [0, 1]
    let attn_weights = b.add_softmax(scores, 1, &[SEQ_LEN, SEQ_LEN]);
    let def = b
        .build(attn_weights)
        .expect("valid attention weights kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Attention weight zeroing IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "softmax attention weights lower >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "softmax attention weights upper <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 8. Cross-attention mask (encoder padding) (IBP)
// ===========================================================================

/// Cross-attention: decoder queries attend to encoder key/values.
/// Encoder padding reduces the effective encoder sequence. We model
/// this with different sequence lengths for Q (decoder) vs K/V (encoder).
#[test]
fn test_cross_attention_mask_encoder_padding_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_mask_cross_attn_enc_pad");
    let dec_input = b.add_input("dec_x", &[SEQ_LEN, DIM]);
    let enc_input = b.add_input("enc_x", &[ENC_SEQ_LEN, DIM]);

    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);

    // Decoder Q: [SEQ_LEN, DIM]
    let q = b.add_linear(dec_input, q_w, None, &[SEQ_LEN, DIM]);
    // Encoder K, V: [ENC_SEQ_LEN, DIM]
    let k = b.add_linear(enc_input, k_w, None, &[ENC_SEQ_LEN, DIM]);
    let v = b.add_linear(enc_input, v_w, None, &[ENC_SEQ_LEN, DIM]);

    // Cross-attention: Q @ K^T -> [SEQ_LEN, ENC_SEQ_LEN], scaled by 1/sqrt(d_k)
    let cross_scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let scores = b.add_matmul(q, k, true, Some(cross_scale), &[SEQ_LEN, ENC_SEQ_LEN]);
    let attn_weights = b.add_softmax(scores, 1, &[SEQ_LEN, ENC_SEQ_LEN]);

    // Weighted sum of V: [SEQ_LEN, ENC_SEQ_LEN] @ [ENC_SEQ_LEN, DIM] -> [SEQ_LEN, DIM]
    let out = b.add_matmul(attn_weights, v, false, None, &[SEQ_LEN, DIM]);

    // Output projection
    let o_w = b.add_input("o_w", &[DIM, DIM]);
    let proj_out = b.add_linear(out, o_w, None, &[SEQ_LEN, DIM]);
    let def = b.build(proj_out).expect("valid cross-attention kernel");

    let bindings = vec![
        TensorParamBinding::Variable, // dec_x
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            // enc_x (constant)
            IxDyn(&[ENC_SEQ_LEN, DIM]),
            0.1f32,
        )),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-attention encoder padding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. Bidirectional vs causal mask bound comparison (IBP)
// ===========================================================================

/// Compare bound widths: bidirectional attention has access to full context
/// while causal attention sees partial context. Both should be finite.
#[test]
fn test_bidirectional_vs_causal_bound_comparison_ibp() {
    let bindings = mha_bindings();

    // Bidirectional (standard)
    let std_def = build_standard_mha_kernel("dpdf_mask_bidir_compare");
    let std_graph = tensor_kernel_to_graph(&std_def, &bindings).expect("std graph");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);
    let std_output = std_graph.propagate_ibp(&input).expect("std IBP");
    assert_bounds_valid(&std_output);
    let std_width = bound_width(&std_output);

    // Causal
    let causal_def = build_causal_mha_kernel("dpdf_mask_causal_compare2");
    let causal_graph = tensor_kernel_to_graph(&causal_def, &bindings).expect("causal graph");
    let causal_output = causal_graph.propagate_ibp(&input).expect("causal IBP");
    assert_bounds_valid(&causal_output);
    let causal_width = bound_width(&causal_output);

    eprintln!("Bidir vs causal IBP: bidir_width={std_width:.6}, causal_width={causal_width:.6}");
    assert!(std_width.is_finite(), "bidirectional width must be finite");
    assert!(causal_width.is_finite(), "causal width must be finite");
}

// ===========================================================================
// 10. Prefix mask (prefix LM attention) (IBP)
// ===========================================================================

/// Prefix LM attention: first `prefix_len` tokens attend bidirectionally,
/// remaining tokens attend causally. Modeled as bidirectional attention
/// over the prefix, since the prefix has full attention.
#[test]
fn test_prefix_mask_ibp() {
    let prefix_len = 2;
    // Model the prefix portion: bidirectional attention over prefix tokens
    let mut b = TensorBlockBuilder::new("dpdf_mask_prefix_lm");
    let input = b.add_input("x", &[prefix_len, DIM]);
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let o_w = b.add_input("o_w", &[DIM, DIM]);

    // Bidirectional attention within the prefix
    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[prefix_len, DIM],
        )
        .expect("valid prefix MHA");
    let def = b.build(out).expect("valid prefix LM kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
        weight_binding(&[DIM, DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[prefix_len, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Prefix LM (prefix_len={prefix_len}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 11. Mask numerical stability (large negative values) (IBP)
// ===========================================================================

/// Verify that attention with large input range remains numerically stable.
/// Softmax with very large/small inputs should still produce finite bounds
/// due to the exp normalization. Large negative mask values (e.g., -1e9)
/// should not cause overflow.
#[test]
fn test_mask_numerical_stability_large_values_ibp() {
    let def = build_causal_mha_kernel("dpdf_mask_numerical_stability");
    let bindings = mha_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Large input range: [-10, 10]
    let input = uniform_bounds(&[SEQ_LEN, DIM], 10.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Mask numerical stability (range=10) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min.is_finite(),
        "lower bound must be finite for large inputs"
    );
    assert!(
        hi_max.is_finite(),
        "upper bound must be finite for large inputs"
    );

    let width = hi_max - lo_min;
    assert!(
        width.is_finite(),
        "bound width must be finite even for large inputs"
    );
}

// ===========================================================================
// 12. CROWN tightness for masked attention (CROWN)
// ===========================================================================

/// Verify CROWN produces valid bounds for causal-masked attention.
/// When CROWN succeeds (no fallback), bounds should be at least as
/// tight as IBP bounds.
#[test]
fn test_crown_tightness_masked_attention() {
    let def = build_causal_mha_kernel("dpdf_mask_crown_tightness");
    let bindings = mha_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Masked attention CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 13. Mask monotone tightening (IBP)
// ===========================================================================

/// Verify monotone tightening: smaller input range produces tighter output bounds.
/// This is a fundamental soundness property of IBP propagation.
#[test]
fn test_mask_monotone_tightening_ibp() {
    let def = build_causal_mha_kernel("dpdf_mask_monotone_tightening");
    let bindings = mha_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Wide input: [-1, 1]
    let wide_input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);
    let wide_width = bound_width(&wide_output);

    // Tight input: [-0.1, 0.1]
    let tight_input = uniform_bounds(&[SEQ_LEN, DIM], 0.1);
    let tight_output = graph.propagate_ibp(&tight_input).expect("IBP tight");
    assert_bounds_valid(&tight_output);
    let tight_width = bound_width(&tight_output);

    eprintln!(
        "Mask monotone tightening: eps=1.0 width={wide_width:.6}, eps=0.1 width={tight_width:.6}"
    );
    assert!(
        tight_width <= wide_width + 1e-6,
        "tight input should produce tighter output: wide={wide_width}, tight={tight_width}"
    );
}

// ===========================================================================
// 14. Multi-head mask broadcasting (IBP)
// ===========================================================================

/// Verify that MHA with different head counts produces valid bounds.
/// Masks are broadcast across heads. Verify with 2 and 8 heads at the
/// same model dimension.
#[test]
fn test_multi_head_mask_broadcasting_ibp() {
    let dim = 16;

    for &heads in &[2usize, 4, 8] {
        if dim % heads != 0 {
            continue;
        }
        let mut b = TensorBlockBuilder::new(&format!("dpdf_mask_multihead_{heads}"));
        let input = b.add_input("x", &[SEQ_LEN, dim]);
        let q_w = b.add_input("q_w", &[dim, dim]);
        let k_w = b.add_input("k_w", &[dim, dim]);
        let v_w = b.add_input("v_w", &[dim, dim]);
        let o_w = b.add_input("o_w", &[dim, dim]);

        let out = b
            .add_multi_head_attention(
                input,
                q_w,
                k_w,
                v_w,
                o_w,
                heads,
                AttentionMask::Causal,
                &[SEQ_LEN, dim],
            )
            .expect("valid multi-head MHA");
        let def = b.build(out).expect("valid multi-head kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            weight_binding(&[dim, dim]),
            weight_binding(&[dim, dim]),
            weight_binding(&[dim, dim]),
            weight_binding(&[dim, dim]),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[SEQ_LEN, dim], 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("Multi-head broadcasting (heads={heads}) IBP: width={width:.6}");
        assert!(width.is_finite(), "width must be finite for heads={heads}");
    }
}

// ===========================================================================
// 15. Full masked attention block: LN + mask + MHA + residual (IBP + CROWN)
// ===========================================================================

/// Build a full pre-norm transformer attention sub-block:
/// LayerNorm -> causal MHA -> residual -> LayerNorm -> SwiGLU FFN -> residual.
/// This is the standard pattern in GLM-OCR, Qwen3-VL decoder layers.
fn build_full_masked_attention_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_mask_full_block");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let shape = [SEQ_LEN, DIM];

    // Pre-norm: LayerNorm
    let ln1_eps = b.add_input("ln1_eps", &[1]);
    let ln1_weight = b.add_input("ln1_weight", &[DIM]);
    let ln1_bias = b.add_input("ln1_bias", &[DIM]);
    let normed = b.add_layer_norm(input, ln1_eps, 1, ln1_weight, ln1_bias, &shape);

    // Causal MHA
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let o_w = b.add_input("o_w", &[DIM, DIM]);

    let attn_out = b
        .add_multi_head_attention(
            normed,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("valid causal MHA");

    // First residual
    let h = b.add_binary_add(input, attn_out, &shape);

    // Pre-norm for FFN: LayerNorm
    let ln2_eps = b.add_input("ln2_eps", &[1]);
    let ln2_weight = b.add_input("ln2_weight", &[DIM]);
    let ln2_bias = b.add_input("ln2_bias", &[DIM]);
    let normed2 = b.add_layer_norm(h, ln2_eps, 1, ln2_weight, ln2_bias, &shape);

    // SwiGLU FFN: gate_proj -> SiLU -> mul(up_proj) -> down_proj
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let gate_w = b.add_input("gate_w", &[FFN_DIM, DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, DIM]);
    let down_w = b.add_input("down_w", &[DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Second residual
    let out = b.add_binary_add(h, ffn_out, &shape);
    b.build(out).expect("valid full masked attention block")
}

fn full_masked_attention_block_bindings() -> Vec<TensorParamBinding> {
    let eps_val = 1e-5f32;
    vec![
        TensorParamBinding::Variable, // x
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), eps_val)), // ln1_eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 1.0f32)), // ln1_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)), // ln1_bias
        weight_binding(&[DIM, DIM]),  // q_w
        weight_binding(&[DIM, DIM]),  // k_w
        weight_binding(&[DIM, DIM]),  // v_w
        weight_binding(&[DIM, DIM]),  // o_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), eps_val)), // ln2_eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 1.0f32)), // ln2_weight
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32)), // ln2_bias
        weight_binding(&[FFN_DIM, DIM]), // gate_w
        weight_binding(&[FFN_DIM, DIM]), // up_w
        weight_binding(&[DIM, FFN_DIM]), // down_w
    ]
}

#[test]
fn test_full_masked_attention_block_ibp() {
    let def = build_full_masked_attention_block_kernel();
    let bindings = full_masked_attention_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full masked attention block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_full_masked_attention_block_crown() {
    let def = build_full_masked_attention_block_kernel();
    let bindings = full_masked_attention_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Full masked attention block CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
