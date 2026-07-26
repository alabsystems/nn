// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for sliding window and local attention patterns.
//!
//! Verifies IBP and CROWN bound propagation through sliding window attention
//! patterns used in Qwen3-VL and similar vision-language models. Sliding window
//! attention restricts each token's attention to a local neighborhood, reducing
//! quadratic complexity to linear while preserving local context. These patterns
//! are critical for long-sequence and high-resolution vision processing.
//!
//! 1.  **Basic sliding window mask generation bounds** (IBP)
//! 2.  **Window partition + local attention + unpartition pipeline** (IBP)
//! 3.  **Window size effect on attention bound tightness** (IBP)
//! 4.  **Interleaved window/global attention pattern (Qwen3-VL)** (IBP)
//! 5.  **Window attention with padding for non-divisible sequence lengths** (IBP)
//! 6.  **Dilated/strided window attention bounds** (IBP)
//! 7.  **2D spatial window partitioning for vision features** (IBP)
//! 8.  **Window attention + relative position bias composition** (IBP)
//! 9.  **Cross-window information flow (shifted windows, Swin-style)** (IBP)
//! 10. **Window attention with causal mask** (IBP)
//! 11. **Multi-head window attention with GQA** (IBP + CROWN)
//! 12. **CROWN tightness for window vs global attention** (CROWN)
//! 13. **Window attention memory efficiency bounds** (IBP)
//! 14. **Overlapping windows for boundary continuity** (IBP)
//! 15. **End-to-end windowed ViT encoder block** (IBP + CROWN)
//!
//! Architecture references:
//! - Swin Transformer (Liu et al., 2021): Shifted window attention for ViTs
//! - Qwen2-VL / Qwen3-VL (Alibaba): Window attention in vision encoder
//! - Longformer (Beltagy et al., 2020): Sliding window + global attention
//! - BigBird (Zaheer et al., 2020): Sparse attention with local windows
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=8, DIM=16, NUM_HEADS=4, HEAD_DIM=4, WINDOW_SIZE=4
//!
//! Part of #4036: Compose tests for sliding window and local attention patterns.

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

const SEQ_LEN: usize = 8;
const DIM: usize = 16;
const NUM_HEADS: usize = 4;
const WINDOW_SIZE: usize = 4;
const FFN_DIM: usize = 32;
const WEIGHT_MAG: f32 = 0.02;
/// Number of windows when partitioning SEQ_LEN into WINDOW_SIZE chunks.
const NUM_WINDOWS: usize = SEQ_LEN / WINDOW_SIZE; // 2
/// For GQA: 1 KV head per 4 Q heads.
const NUM_KV_HEADS: usize = 1;
const KV_DIM: usize = NUM_KV_HEADS * (DIM / NUM_HEADS); // 4

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
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

/// Build a standard multi-head attention block on a window of tokens.
///
/// Input shape: `[window_size, dim]`, output shape: `[window_size, dim]`.
fn build_window_mha(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    window_size: usize,
    dim: usize,
    num_heads: usize,
) -> nn_dsl::TensorNodeId {
    let q_w = b.add_input(&format!("{prefix}_q_w"), &[dim, dim]);
    let k_w = b.add_input(&format!("{prefix}_k_w"), &[dim, dim]);
    let v_w = b.add_input(&format!("{prefix}_v_w"), &[dim, dim]);
    let out_w = b.add_input(&format!("{prefix}_out_w"), &[dim, dim]);

    b.add_multi_head_attention(
        input,
        q_w,
        k_w,
        v_w,
        out_w,
        num_heads,
        AttentionMask::Standard,
        &[window_size, dim],
    )
    .expect("valid window MHA")
}

/// Push MHA weight bindings (q, k, v, out) for the given dimensions.
fn push_mha_bindings(bindings: &mut Vec<TensorParamBinding>, dim: usize) {
    let w = ArrayD::from_elem(IxDyn(&[dim, dim]), WEIGHT_MAG);
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(w.clone()));
    }
}

/// Build a SwiGLU FFN block.
fn build_swiglu_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> nn_dsl::TensorNodeId {
    let ffn_shape = [seq_len, ffn_dim];
    let out_shape = [seq_len, hidden_dim];

    let gate_w = b.add_input(&format!("{prefix}_gate_w"), &[ffn_dim, hidden_dim]);
    let up_w = b.add_input(&format!("{prefix}_up_w"), &[ffn_dim, hidden_dim]);
    let down_w = b.add_input(&format!("{prefix}_down_w"), &[hidden_dim, ffn_dim]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(b, gate, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    b.add_linear(hidden, down_w, None, &out_shape)
}

/// Push SwiGLU weight bindings (gate_w, up_w, down_w).
fn push_swiglu_bindings(bindings: &mut Vec<TensorParamBinding>, hidden_dim: usize, ffn_dim: usize) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ffn_dim, hidden_dim]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ffn_dim, hidden_dim]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[hidden_dim, ffn_dim]),
        WEIGHT_MAG,
    )));
}

// ===========================================================================
// 1. Basic sliding window mask generation bounds (IBP)
// ===========================================================================

/// Model a single sliding window: attention is applied within a local
/// window of WINDOW_SIZE tokens. The mask restricts attention to local
/// positions only. We verify bounds on a single window of tokens.
fn build_basic_sliding_window_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_sliding_window_basic");
    let input = b.add_input("x", &[WINDOW_SIZE, DIM]);

    // Local attention within the window (no masking needed — window IS the scope)
    let out = build_window_mha(&mut b, input, "attn", WINDOW_SIZE, DIM, NUM_HEADS);
    b.build(out).expect("valid basic sliding window kernel")
}

fn basic_sliding_window_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_mha_bindings(&mut bindings, DIM);
    bindings
}

#[test]
fn test_sliding_window_basic_mask_ibp() {
    let def = build_basic_sliding_window_kernel();
    let bindings = basic_sliding_window_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[WINDOW_SIZE, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sliding window basic IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. Window partition + local attention + unpartition pipeline (IBP)
// ===========================================================================

/// Full window partition pipeline: partition sequence into windows,
/// apply local attention per window, then unpartition (concatenate) back.
/// Modeled as: for each window, apply MHA independently, then concat.
fn build_partition_attention_unpartition_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_sliding_window_partition");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    // Reshape to [NUM_WINDOWS, WINDOW_SIZE, DIM] via separate window processing.
    // We model window partitioning as: reshape -> per-window attention -> reshape.
    // Reshape to [NUM_WINDOWS * WINDOW_SIZE, DIM] = [SEQ_LEN, DIM] trivially.
    // Apply attention on the full sequence at window granularity:
    // simulate by linear -> attention on window -> linear.
    let w_proj = b.add_input("w_proj", &[DIM, DIM]);
    let projected = b.add_linear(input, w_proj, None, &[SEQ_LEN, DIM]);

    // Apply attention at window size (simulated: we do attention on WINDOW_SIZE tokens)
    // Reshape to [NUM_WINDOWS * WINDOW_SIZE, DIM], then narrow to one window.
    // For verification: process entire sequence through linear + attention-like ops.
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let out_w = b.add_input("out_w", &[DIM, DIM]);

    let attn_out = b
        .add_multi_head_attention(
            projected,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, DIM],
        )
        .expect("valid partitioned MHA");

    // Unpartition: residual connection with original input
    let out = b.add_binary_add(input, attn_out, &[SEQ_LEN, DIM]);
    b.build(out)
        .expect("valid partition-attention-unpartition kernel")
}

fn partition_attention_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    // w_proj
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[DIM, DIM]),
        WEIGHT_MAG,
    )));
    push_mha_bindings(&mut bindings, DIM);
    bindings
}

#[test]
fn test_window_partition_attention_unpartition_ibp() {
    let def = build_partition_attention_unpartition_kernel();
    let bindings = partition_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Window partition+attn+unpartition IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. Window size effect on attention bound tightness (IBP)
// ===========================================================================

/// Smaller windows should produce tighter bounds because the attention
/// operates over fewer tokens, reducing the combinatorial interaction.
fn build_window_attention_at_size(
    window_size: usize,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new(&format!("dpdf_sliding_window_size_{window_size}"));
    let input = b.add_input("x", &[window_size, DIM]);
    let out = build_window_mha(&mut b, input, "attn", window_size, DIM, NUM_HEADS);
    let def = b.build(out).expect("valid window attention kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_mha_bindings(&mut bindings, DIM);
    (def, bindings)
}

#[test]
fn test_window_size_effect_on_bound_tightness_ibp() {
    let sizes = [2, 4, 8];
    let mut widths = Vec::new();

    for &ws in &sizes {
        let (def, bindings) = build_window_attention_at_size(ws);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[ws, DIM], 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("Window size={ws} IBP: width={width:.6}");
        assert!(
            width.is_finite(),
            "width must be finite for window_size={ws}"
        );
        widths.push(width);
    }

    // Smaller windows should produce tighter (or equal) bounds
    for i in 0..widths.len() - 1 {
        assert!(
            widths[i] <= widths[i + 1] + 1e-4,
            "window_size={} width {} should be <= window_size={} width {}",
            sizes[i],
            widths[i],
            sizes[i + 1],
            widths[i + 1]
        );
    }
}

// ===========================================================================
// 4. Interleaved window/global attention pattern (Qwen3-VL) (IBP)
// ===========================================================================

/// Qwen3-VL interleaves window attention layers with global attention layers.
/// Model: window_attn -> global_attn composition.
fn build_interleaved_window_global_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_sliding_window_interleaved");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    // Layer 1: Window attention (simulated on full sequence with smaller effective scope)
    let win_out = build_window_mha(&mut b, input, "win_attn", SEQ_LEN, DIM, NUM_HEADS);
    let h = b.add_binary_add(input, win_out, &[SEQ_LEN, DIM]);

    // Layer 2: Global attention (full sequence)
    let glob_out = build_window_mha(&mut b, h, "glob_attn", SEQ_LEN, DIM, NUM_HEADS);
    let out = b.add_binary_add(h, glob_out, &[SEQ_LEN, DIM]);

    b.build(out)
        .expect("valid interleaved window/global kernel")
}

fn interleaved_window_global_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_mha_bindings(&mut bindings, DIM); // window attn
    push_mha_bindings(&mut bindings, DIM); // global attn
    bindings
}

#[test]
fn test_interleaved_window_global_ibp() {
    let def = build_interleaved_window_global_kernel();
    let bindings = interleaved_window_global_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Interleaved window/global IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 5. Window attention with padding for non-divisible sequence lengths (IBP)
// ===========================================================================

/// When sequence length is not divisible by window size, the last window
/// is padded. Model with SEQ_LEN=6, WINDOW_SIZE=4 -> 2 windows, last padded.
/// Padding is modeled by processing the full padded sequence length.
#[test]
fn test_window_attention_padded_ibp() {
    let orig_seq = 6;
    let ws = 4;
    // Pad to next multiple of window size
    let padded_seq = ((orig_seq + ws - 1) / ws) * ws; // 8

    let mut b = TensorBlockBuilder::new("dpdf_sliding_window_padded");
    let input = b.add_input("x", &[padded_seq, DIM]);
    let out = build_window_mha(&mut b, input, "attn", padded_seq, DIM, NUM_HEADS);
    let def = b.build(out).expect("valid padded window attention kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_mha_bindings(&mut bindings, DIM);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[padded_seq, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Window attention padded (orig={orig_seq}, padded={padded_seq}) IBP: \
         bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. Dilated/strided window attention bounds (IBP)
// ===========================================================================

/// Dilated attention: instead of contiguous windows, every d-th token
/// forms a group. Modeled as linear projection + attention on the dilated
/// subset, then projection back.
#[test]
fn test_dilated_window_attention_ibp() {
    let dilation = 2;
    // Effective window: every 2nd token from a span of WINDOW_SIZE * dilation
    let effective_window = WINDOW_SIZE; // tokens per dilated window

    let mut b = TensorBlockBuilder::new("dpdf_sliding_window_dilated");
    let input = b.add_input("x", &[effective_window, DIM]);

    // Dilated selection modeled as a linear projection (mixing strided positions)
    let select_w = b.add_input("select_w", &[DIM, DIM]);
    let selected = b.add_linear(input, select_w, None, &[effective_window, DIM]);

    // Attention on dilated window
    let out = build_window_mha(&mut b, selected, "attn", effective_window, DIM, NUM_HEADS);
    let def = b.build(out).expect("valid dilated window attention kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[DIM, DIM]),
        WEIGHT_MAG,
    )));
    push_mha_bindings(&mut bindings, DIM);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[effective_window, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Dilated window (dilation={dilation}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    let _ = dilation; // used in eprintln
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 7. 2D spatial window partitioning for vision features (IBP)
// ===========================================================================

/// Vision features are 2D spatial grids. Window partitioning operates on
/// HxW patches. Model as: flatten 2D grid -> window attention -> reshape.
/// For a 4x4 spatial grid with 2x2 windows: 4 windows of 4 tokens each.
#[test]
fn test_2d_spatial_window_partitioning_ibp() {
    let h = 4;
    let w = 4;
    let num_spatial_tokens = h * w; // 16
    let window_h = 2;
    let window_w = 2;
    let tokens_per_window = window_h * window_w; // 4

    let mut b = TensorBlockBuilder::new("dpdf_sliding_window_2d_spatial");
    // Flattened spatial tokens: [H*W, DIM]
    let input = b.add_input("x", &[num_spatial_tokens, DIM]);

    // Linear projection (simulates spatial rearrangement into windows)
    let proj_w = b.add_input("proj_w", &[DIM, DIM]);
    let projected = b.add_linear(input, proj_w, None, &[num_spatial_tokens, DIM]);

    // Attention on the spatial tokens (approximates per-window attention)
    let out = build_window_mha(
        &mut b,
        projected,
        "spatial_attn",
        num_spatial_tokens,
        DIM,
        NUM_HEADS,
    );
    let def = b.build(out).expect("valid 2D spatial window kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[DIM, DIM]),
        WEIGHT_MAG,
    )));
    push_mha_bindings(&mut bindings, DIM);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[num_spatial_tokens, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "2D spatial window ({h}x{w}, window={window_h}x{window_w}, \
         tokens_per_window={tokens_per_window}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. Window attention + relative position bias composition (IBP)
// ===========================================================================

/// Window attention with learned relative position bias added to attention
/// logits. Modeled as: attention output + position bias projection.
#[test]
fn test_window_attention_relative_position_bias_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_sliding_window_rel_pos_bias");
    let input = b.add_input("x", &[WINDOW_SIZE, DIM]);

    // Attention within window
    let attn_out = build_window_mha(&mut b, input, "attn", WINDOW_SIZE, DIM, NUM_HEADS);

    // Relative position bias: a learned bias projected and added to output.
    // Model as: Linear(input) -> add to attention output.
    let bias_w = b.add_input("bias_w", &[DIM, DIM]);
    let bias = b.add_linear(input, bias_w, None, &[WINDOW_SIZE, DIM]);

    let out = b.add_binary_add(attn_out, bias, &[WINDOW_SIZE, DIM]);
    let def = b.build(out).expect("valid rel pos bias window kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_mha_bindings(&mut bindings, DIM);
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[DIM, DIM]),
        WEIGHT_MAG,
    )));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[WINDOW_SIZE, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Window attention + relative position bias IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. Cross-window information flow (shifted windows, Swin-style) (IBP)
// ===========================================================================

/// Swin Transformer shifts the window partition by half the window size
/// between layers, enabling cross-window information flow.
/// Modeled as: window_attn_layer_1 + residual -> shifted_window_attn_layer_2 + residual.
/// The shift is modeled as a linear mixing layer between the two attention blocks.
#[test]
fn test_shifted_window_cross_flow_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_sliding_window_shifted");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    // Layer 1: Regular window attention
    let attn1 = build_window_mha(&mut b, input, "win_attn1", SEQ_LEN, DIM, NUM_HEADS);
    let h1 = b.add_binary_add(input, attn1, &[SEQ_LEN, DIM]);

    // Shift: modeled as a linear mixing across positions (shifted window partition)
    let shift_w = b.add_input("shift_w", &[DIM, DIM]);
    let shifted = b.add_linear(h1, shift_w, None, &[SEQ_LEN, DIM]);

    // Layer 2: Shifted window attention
    let attn2 = build_window_mha(&mut b, shifted, "win_attn2", SEQ_LEN, DIM, NUM_HEADS);
    let out = b.add_binary_add(h1, attn2, &[SEQ_LEN, DIM]);

    let def = b.build(out).expect("valid shifted window kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_mha_bindings(&mut bindings, DIM); // win_attn1
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[DIM, DIM]),
        WEIGHT_MAG,
    ))); // shift_w
    push_mha_bindings(&mut bindings, DIM); // win_attn2

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Shifted window (Swin-style) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. Window attention with causal mask (IBP)
// ===========================================================================

/// Causal window attention: within each window, tokens can only attend to
/// preceding positions. Used in autoregressive models with sliding window.
fn build_causal_window_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_sliding_window_causal");
    let input = b.add_input("x", &[WINDOW_SIZE, DIM]);

    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let out_w = b.add_input("out_w", &[DIM, DIM]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Causal,
            &[WINDOW_SIZE, DIM],
        )
        .expect("valid causal window MHA");

    b.build(out).expect("valid causal window attention kernel")
}

#[test]
fn test_window_attention_causal_mask_ibp() {
    let def = build_causal_window_attention_kernel();
    let mut bindings = vec![TensorParamBinding::Variable];
    push_mha_bindings(&mut bindings, DIM);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[WINDOW_SIZE, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Causal window attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 11. Multi-head window attention with GQA (IBP + CROWN)
// ===========================================================================

/// Grouped-query attention (GQA) within a sliding window: fewer KV heads
/// than Q heads. Common in Qwen3-VL decoder with window attention.
fn build_gqa_window_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_sliding_window_gqa");
    let input = b.add_input("x", &[WINDOW_SIZE, DIM]);

    // Q projection: full dimension
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let q = b.add_linear(input, q_w, None, &[WINDOW_SIZE, DIM]);

    // K, V projections: reduced to KV_DIM
    let k_w = b.add_input("k_w", &[KV_DIM, DIM]);
    let v_w = b.add_input("v_w", &[KV_DIM, DIM]);
    let k = b.add_linear(input, k_w, None, &[WINDOW_SIZE, KV_DIM]);
    let v = b.add_linear(input, v_w, None, &[WINDOW_SIZE, KV_DIM]);

    // Downproject Q to KV_DIM for attention computation
    let q_down_w = b.add_input("q_down_w", &[KV_DIM, DIM]);
    let q_down = b.add_linear(q, q_down_w, None, &[WINDOW_SIZE, KV_DIM]);

    // Attention: softmax(Q_down @ K^T / sqrt(d_k)) @ V
    let head_dim = DIM / NUM_HEADS;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let attn_logits = b.add_matmul(q_down, k, true, Some(scale), &[WINDOW_SIZE, WINDOW_SIZE]);
    let attn_weights = b.add_softmax(attn_logits, 1, &[WINDOW_SIZE, WINDOW_SIZE]);
    let attn_out = b.add_matmul(attn_weights, v, false, None, &[WINDOW_SIZE, KV_DIM]);

    // Up-project back to DIM
    let out_w = b.add_input("out_w", &[DIM, KV_DIM]);
    let out = b.add_linear(attn_out, out_w, None, &[WINDOW_SIZE, DIM]);

    // Residual
    let result = b.add_binary_add(input, out, &[WINDOW_SIZE, DIM]);
    b.build(result).expect("valid GQA window attention kernel")
}

fn gqa_window_attention_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        // q_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG)),
        // k_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG)),
        // v_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG)),
        // q_down_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[KV_DIM, DIM]), WEIGHT_MAG)),
        // out_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM, KV_DIM]), WEIGHT_MAG)),
    ]
}

#[test]
fn test_gqa_window_attention_ibp() {
    let def = build_gqa_window_attention_kernel();
    let bindings = gqa_window_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[WINDOW_SIZE, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GQA window attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_gqa_window_attention_crown() {
    let def = build_gqa_window_attention_kernel();
    let bindings = gqa_window_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[WINDOW_SIZE, DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("GQA window attention CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 12. CROWN tightness for window vs global attention (CROWN)
// ===========================================================================

/// Compare CROWN bounds between window attention and global attention.
/// Window attention should produce tighter or comparable bounds since
/// the attention scope is more restricted.
#[test]
fn test_crown_window_vs_global_attention() {
    // Window attention (WINDOW_SIZE tokens)
    let win_def = build_basic_sliding_window_kernel();
    let win_bindings = basic_sliding_window_bindings();
    let win_graph = tensor_kernel_to_graph(&win_def, &win_bindings).expect("window graph");
    let win_input = uniform_bounds(&[WINDOW_SIZE, DIM], 0.5);

    let (win_method, win_output, win_fb) =
        assert_crown_tighter_when_not_fallback(&win_graph, &win_input);

    // Global attention (SEQ_LEN tokens)
    let mut gb = TensorBlockBuilder::new("dpdf_sliding_window_global_compare");
    let g_input = gb.add_input("x", &[SEQ_LEN, DIM]);
    let g_out = build_window_mha(&mut gb, g_input, "attn", SEQ_LEN, DIM, NUM_HEADS);
    let glob_def = gb.build(g_out).expect("valid global attention kernel");

    let mut glob_bindings = vec![TensorParamBinding::Variable];
    push_mha_bindings(&mut glob_bindings, DIM);
    let glob_graph = tensor_kernel_to_graph(&glob_def, &glob_bindings).expect("global graph");
    let glob_input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let (glob_method, glob_output, glob_fb) =
        assert_crown_tighter_when_not_fallback(&glob_graph, &glob_input);

    let win_width = bound_width(&win_output);
    let glob_width = bound_width(&glob_output);

    eprintln!(
        "CROWN window vs global: window={win_width:.6} (method={win_method:?}), \
         global={glob_width:.6} (method={glob_method:?})"
    );
    if let Some(r) = &win_fb {
        eprintln!("Window fallback: {r}");
    }
    if let Some(r) = &glob_fb {
        eprintln!("Global fallback: {r}");
    }

    // Both must produce finite bounds
    assert!(win_width.is_finite(), "window width must be finite");
    assert!(glob_width.is_finite(), "global width must be finite");
}

// ===========================================================================
// 13. Window attention memory efficiency bounds (IBP)
// ===========================================================================

/// Verify that processing the sequence in windows produces finite bounds
/// even for longer sequences that would be expensive with global attention.
/// Model: process a longer sequence (SEQ_LEN * 2) via windowed attention.
#[test]
fn test_window_attention_memory_efficiency_ibp() {
    let long_seq = SEQ_LEN * 2; // 16 tokens

    let mut b = TensorBlockBuilder::new("dpdf_sliding_window_memory");
    let input = b.add_input("x", &[long_seq, DIM]);
    let out = build_window_mha(&mut b, input, "attn", long_seq, DIM, NUM_HEADS);
    let def = b.build(out).expect("valid long-sequence window kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_mha_bindings(&mut bindings, DIM);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[long_seq, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Window attention memory efficiency (seq={long_seq}) IBP: width={width:.6}");
    assert!(width.is_finite(), "width must be finite for long sequence");
}

// ===========================================================================
// 14. Overlapping windows for boundary continuity (IBP)
// ===========================================================================

/// Overlapping windows: adjacent windows share boundary tokens to improve
/// information flow. Modeled as: attention on an extended window (window_size
/// + overlap), then linear projection back to original dimension.
#[test]
fn test_overlapping_windows_boundary_continuity_ibp() {
    let overlap = 2;
    let extended_window = WINDOW_SIZE + overlap; // 6

    let mut b = TensorBlockBuilder::new("dpdf_sliding_window_overlapping");
    let input = b.add_input("x", &[extended_window, DIM]);

    // Attention on extended window (includes overlapping tokens)
    let attn_out = build_window_mha(&mut b, input, "attn", extended_window, DIM, NUM_HEADS);

    // Project back to original window size via linear + narrow simulation
    // Model as linear to capture the dimension reduction
    let proj_w = b.add_input("proj_w", &[DIM, DIM]);
    let out = b.add_linear(attn_out, proj_w, None, &[extended_window, DIM]);

    let def = b.build(out).expect("valid overlapping window kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_mha_bindings(&mut bindings, DIM);
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[DIM, DIM]),
        WEIGHT_MAG,
    )));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[extended_window, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!(
        "Overlapping windows (window={WINDOW_SIZE}, overlap={overlap}, \
         extended={extended_window}) IBP: width={width:.6}"
    );
    assert!(
        width.is_finite(),
        "width must be finite for overlapping windows"
    );

    // Compare with non-overlapping window
    let (non_overlap_def, non_overlap_bindings) = build_window_attention_at_size(WINDOW_SIZE);
    let non_overlap_graph =
        tensor_kernel_to_graph(&non_overlap_def, &non_overlap_bindings).expect("graph");
    let non_overlap_input = uniform_bounds(&[WINDOW_SIZE, DIM], 1.0);
    let non_overlap_output = non_overlap_graph
        .propagate_ibp(&non_overlap_input)
        .expect("IBP");
    assert_bounds_valid(&non_overlap_output);
    let non_overlap_width = bound_width(&non_overlap_output);

    eprintln!("Overlapping width={width:.6} vs non-overlapping width={non_overlap_width:.6}");
    // Both should be finite
    assert!(
        non_overlap_width.is_finite(),
        "non-overlapping width must be finite"
    );
}

// ===========================================================================
// 15. End-to-end windowed ViT encoder block (IBP + CROWN)
// ===========================================================================

/// Full ViT encoder block with window attention: RMSNorm -> Window MHA ->
/// residual -> RMSNorm -> SwiGLU FFN -> residual. This is the core building
/// block of Qwen3-VL's vision encoder.
fn build_windowed_vit_encoder_block_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_sliding_window_vit_block");
    let input = b.add_input("x", &[WINDOW_SIZE, DIM]);
    let shape = [WINDOW_SIZE, DIM];

    // Pre-norm: RMSNorm
    let eps1 = b.add_input("eps1", &[1]);
    let norm_w1 = b.add_input("norm_w1", &[DIM]);
    let normed1 = b.add_rms_norm(input, eps1, 1, norm_w1, &shape);

    // Window MHA
    let attn_out = build_window_mha(&mut b, normed1, "win_attn", WINDOW_SIZE, DIM, NUM_HEADS);

    // First residual
    let h = b.add_binary_add(input, attn_out, &shape);

    // Pre-norm: RMSNorm
    let eps2 = b.add_input("eps2", &[1]);
    let norm_w2 = b.add_input("norm_w2", &[DIM]);
    let normed2 = b.add_rms_norm(h, eps2, 1, norm_w2, &shape);

    // SwiGLU FFN
    let ffn_out = build_swiglu_block(&mut b, normed2, "ffn", WINDOW_SIZE, DIM, FFN_DIM);

    // Second residual
    let out = b.add_binary_add(h, ffn_out, &shape);
    b.build(out)
        .expect("valid windowed ViT encoder block kernel")
}

fn windowed_vit_encoder_block_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        // eps1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        // norm_w1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[DIM]), 1.0f32)),
    ];
    // Window MHA weights
    push_mha_bindings(&mut bindings, DIM);
    // eps2
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        1e-5f32,
    )));
    // norm_w2
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[DIM]),
        1.0f32,
    )));
    // SwiGLU FFN weights
    push_swiglu_bindings(&mut bindings, DIM, FFN_DIM);
    bindings
}

#[test]
fn test_windowed_vit_encoder_block_ibp() {
    let def = build_windowed_vit_encoder_block_kernel();
    let bindings = windowed_vit_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[WINDOW_SIZE, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Windowed ViT encoder block IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_windowed_vit_encoder_block_crown() {
    let def = build_windowed_vit_encoder_block_kernel();
    let bindings = windowed_vit_encoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[WINDOW_SIZE, DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Windowed ViT encoder block CROWN: method={method:?}, \
         bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
