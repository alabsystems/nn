// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Vision encoder depth NY composition.
//!
//! Verifies IBP and CROWN bounds propagation through ViT encoder stacks
//! at increasing depths — from 1-block to 8-block — with variants for
//! pre-norm / post-norm, window attention, cross-attention, and
//! model-specific depth profiles (SigLIP2, Qwen3-VL, SVTR).
//!
//! 1. **1-block encoder (IBP + CROWN)**: LN -> MHA -> residual -> LN -> FFN -> residual.
//! 2. **2-block encoder stack (IBP + CROWN)**: Bound width after 2 blocks.
//! 3. **4-block encoder stack (IBP)**: Bound width after 4 blocks.
//! 4. **8-block encoder stack (IBP)**: Bound width after 8 blocks.
//! 5. **Bound width vs depth curve (IBP)**: Monotone widening tracked.
//! 6. **Pre-norm encoder (IBP)**: RMSNorm -> attention -> residual pattern.
//! 7. **Post-norm encoder (IBP)**: Attention -> LayerNorm -> residual pattern.
//! 8. **Encoder with window attention (IBP)**: Local attention depth scaling.
//! 9. **Encoder with cross-attention (IBP)**: Depth impact on cross-attn bounds.
//! 10. **SigLIP2 encoder depth (IBP)**: Granite-Docling ViT depth profile.
//! 11. **Qwen3-VL encoder depth (IBP)**: Window ViT depth profile.
//! 12. **SVTR encoder depth (IBP)**: PaddleOCR recognition encoder.
//! 13. **Depth vs CROWN tightness (CROWN)**: CROWN advantage at increasing depth.
//! 14. **Encoder depth monotone (IBP)**: Deeper -> wider bounds (verified property).
//! 15. **Encoder depth + head (IBP)**: Full encoder -> projection -> sigmoid.
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=64, FFN_DIM=128, NUM_HEADS=4
//!
//! Part of #4009: Compose tests for vision encoder depth bound tracking.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Hidden dimension for encoder layers.
const HIDDEN_DIM: usize = 64;
/// FFN intermediate dimension.
const FFN_DIM: usize = 128;
/// Sequence length for [SEQ_LEN, HIDDEN_DIM] inputs.
const SEQ_LEN: usize = 4;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 16
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Encoder memory sequence length for cross-attention tests.
const ENC_MEM_LEN: usize = 8;
/// Output projection size for head tests.
const PROJ_DIM: usize = 32;

// ---------------------------------------------------------------------------
// Helper: Build a single pre-norm ViT encoder block (LayerNorm variant)
// ---------------------------------------------------------------------------

/// Append one pre-norm ViT encoder block:
///   LN -> MHA -> residual -> LN -> FFN (Linear -> GELU -> Linear) -> residual.
///
/// `prefix` distinguishes layer parameters across blocks (e.g., "b1_", "b2_").
/// Returns the output node.
fn add_encoder_block_layernorm(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::tensor_ir::TensorNodeId,
    prefix: &str,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention LayerNorm
    let ln1_eps = b.add_input(&format!("{prefix}ln1_eps"), &[1]);
    let ln1_w = b.add_input(&format!("{prefix}ln1_w"), &[HIDDEN_DIM]);
    let ln1_b = b.add_input(&format!("{prefix}ln1_b"), &[HIDDEN_DIM]);
    let normed1 = b.add_layer_norm(input, ln1_eps, 1, ln1_w, ln1_b, &shape);

    // Self-attention: Q/K/V projection + attention + output projection
    let q_w = b.add_input(&format!("{prefix}q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("{prefix}k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("{prefix}v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input(&format!("{prefix}out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // Residual after attention
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN LayerNorm
    let ln2_eps = b.add_input(&format!("{prefix}ln2_eps"), &[1]);
    let ln2_w = b.add_input(&format!("{prefix}ln2_w"), &[HIDDEN_DIM]);
    let ln2_b = b.add_input(&format!("{prefix}ln2_b"), &[HIDDEN_DIM]);
    let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &shape);

    // FFN: Linear -> GELU -> Linear
    let ffn_up_w = b.add_input(&format!("{prefix}ffn_up_w"), &[FFN_DIM, HIDDEN_DIM]);
    let ffn_down_w = b.add_input(&format!("{prefix}ffn_down_w"), &[HIDDEN_DIM, FFN_DIM]);

    let ffn_up = b.add_linear(normed2, ffn_up_w, None, &ffn_shape);
    let ffn_act = b.add_gelu(ffn_up, &ffn_shape);
    let ffn_out = b.add_linear(ffn_act, ffn_down_w, None, &shape);

    // Residual after FFN
    b.add_binary_add(res1, ffn_out, &shape)
}

/// Push one LayerNorm encoder block's bindings (13 params) onto the vec.
fn push_encoder_block_layernorm_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn_up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    // LN1: eps, weight, bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
    // Attention: Q, K, V, output projections
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj_w));
    // LN2: eps, weight, bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ln_w));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b));
    // FFN: up, down
    bindings.push(TensorParamBinding::ConstantTensor(ffn_up_w));
    bindings.push(TensorParamBinding::ConstantTensor(ffn_down_w));
}

/// Build an N-block ViT encoder stack with LayerNorm pre-norm blocks.
fn build_n_block_encoder(num_blocks: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(&format!("enc_depth_{num_blocks}block"));
    let input = b.add_input("patches", &[SEQ_LEN, HIDDEN_DIM]);

    let mut x = input;
    for i in 0..num_blocks {
        x = add_encoder_block_layernorm(&mut b, x, &format!("b{}_", i + 1));
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid {num_blocks}-block encoder kernel: {e}"))
}

/// Build bindings for an N-block ViT encoder stack.
fn n_block_encoder_bindings(num_blocks: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // patches
    for _ in 0..num_blocks {
        push_encoder_block_layernorm_bindings(&mut bindings);
    }
    bindings
}

// ===========================================================================
// 1. 1-block encoder (IBP + CROWN)
// ===========================================================================

/// 1-block encoder IBP: LN -> MHA -> residual -> LN -> FFN -> residual.
#[test]
fn test_encoder_depth_1block_ibp() {
    let def = build_n_block_encoder(1);
    let bindings = n_block_encoder_bindings(1);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 1-block encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "1-block encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("encoder depth 1-block IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// 1-block encoder CROWN linearization.
#[test]
fn test_encoder_depth_1block_crown() {
    let def = build_n_block_encoder(1);
    let bindings = n_block_encoder_bindings(1);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("encoder depth 1-block CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 2. 2-block encoder stack (IBP + CROWN)
// ===========================================================================

/// 2-block encoder IBP: bound width after 2 blocks.
#[test]
fn test_encoder_depth_2block_ibp() {
    let def = build_n_block_encoder(2);
    let bindings = n_block_encoder_bindings(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-block encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "2-block encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("encoder depth 2-block IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// 2-block encoder CROWN linearization.
#[test]
fn test_encoder_depth_2block_crown() {
    let def = build_n_block_encoder(2);
    let bindings = n_block_encoder_bindings(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("encoder depth 2-block CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 3. 4-block encoder stack (IBP)
// ===========================================================================

/// 4-block encoder IBP: bound width after 4 blocks.
#[test]
fn test_encoder_depth_4block_ibp() {
    let def = build_n_block_encoder(4);
    let bindings = n_block_encoder_bindings(4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-block encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "4-block encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("encoder depth 4-block IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 4. 8-block encoder stack (IBP)
// ===========================================================================

/// 8-block encoder IBP: bound width tracking through 8 blocks.
#[test]
fn test_encoder_depth_8block_ibp() {
    let def = build_n_block_encoder(8);
    let bindings = n_block_encoder_bindings(8);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 8-block encoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "8-block encoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("encoder depth 8-block IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(width > 0.0, "non-trivial bound width at 8 blocks");
}

// ===========================================================================
// 5. Bound width vs depth curve: monotone widening tracked (IBP)
// ===========================================================================

/// Track bound width across 1, 2, 4 encoder blocks and verify monotone widening.
///
/// IBP over-approximation accumulates through depth, so bounds should widen
/// (or at minimum stay equal) as we add more blocks.
#[test]
fn test_encoder_depth_bound_width_vs_depth_monotone() {
    let depths = [1usize, 2, 4];
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let mut prev_width: Option<f32> = None;

    for &depth in &depths {
        let def = build_n_block_encoder(depth);
        let bindings = n_block_encoder_bindings(depth);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        let output = graph.propagate_ibp(&input).expect("IBP");

        assert_bounds_valid(&output);
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("encoder depth={depth}: width={width:.6}, bounds=[{lo_min}, {hi_max}]");

        if let Some(prev_w) = prev_width {
            // Monotone widening: deeper stacks produce wider (or equal) bounds.
            // Small tolerance for numerical noise.
            let tolerance = prev_w * 0.01 + 1e-4;
            assert!(
                width >= prev_w - tolerance,
                "depth {depth}: width {width:.6} should be >= previous {prev_w:.6} - tol {tolerance:.6}"
            );
        }
        prev_width = Some(width);
    }
}

// ===========================================================================
// 6. Pre-norm encoder: RMSNorm -> attention -> residual (IBP)
// ===========================================================================

/// Build a 2-block pre-norm encoder with RMSNorm (Granite-Docling / Qwen3-VL style).
///
/// Each block: RMSNorm -> MHA -> residual -> RMSNorm -> SwiGLU FFN -> residual.
fn build_prenorm_rmsnorm_encoder(num_blocks: usize) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new(&format!("enc_depth_prenorm_{num_blocks}b"));
    let input = b.add_input("patches", &shape);

    let mut x = input;
    for i in 0..num_blocks {
        let pfx = format!("b{}_", i + 1);

        // Pre-attention RMSNorm
        let n1_eps = b.add_input(&format!("{pfx}rn1_eps"), &[1]);
        let n1_w = b.add_input(&format!("{pfx}rn1_w"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(x, n1_eps, 1, n1_w, &shape);

        // Self-attention
        let q_w = b.add_input(&format!("{pfx}q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{pfx}k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{pfx}v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{pfx}out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, q_w, None, &shape);
        let k = b.add_linear(normed1, k_w, None, &shape);
        let v = b.add_linear(normed1, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);
        let res1 = b.add_binary_add(x, attn_out, &shape);

        // Pre-FFN RMSNorm
        let n2_eps = b.add_input(&format!("{pfx}rn2_eps"), &[1]);
        let n2_w = b.add_input(&format!("{pfx}rn2_w"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, n2_eps, 1, n2_w, &shape);

        // SwiGLU FFN
        let gate_w = b.add_input(&format!("{pfx}gate_w"), &[FFN_DIM, HIDDEN_DIM]);
        let up_w = b.add_input(&format!("{pfx}up_w"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("{pfx}down_w"), &[HIDDEN_DIM, FFN_DIM]);

        let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
        let gate_sig = b.add_sigmoid(gate, &ffn_shape);
        let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
        let up = b.add_linear(normed2, up_w, None, &ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
        let ffn_out = b.add_linear(hidden, down_w, None, &shape);

        x = b.add_binary_add(res1, ffn_out, &shape);
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid prenorm RMSNorm encoder: {e}"))
}

fn prenorm_rmsnorm_encoder_bindings(num_blocks: usize) -> Vec<TensorParamBinding> {
    let rn_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // patches
    for _ in 0..num_blocks {
        // RMSNorm1: eps, weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(rn_w.clone()));
        // Attention: Q, K, V, out
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        // RMSNorm2: eps, weight
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(rn_w.clone()));
        // SwiGLU FFN: gate, up, down
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone()));
    }
    bindings
}

/// Pre-norm RMSNorm 2-block encoder IBP: Granite-Docling / Qwen3-VL style.
#[test]
fn test_encoder_depth_prenorm_rmsnorm_ibp() {
    let def = build_prenorm_rmsnorm_encoder(2);
    let bindings = prenorm_rmsnorm_encoder_bindings(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through pre-norm RMSNorm encoder");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("encoder depth pre-norm RMSNorm 2-block IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 7. Post-norm encoder: attention -> LayerNorm -> residual (IBP)
// ===========================================================================

/// Build a 2-block post-norm encoder (original Transformer / Table Transformer style).
///
/// Each block: MHA -> LayerNorm(attn_out + input) -> FFN -> LayerNorm(ffn_out + mid).
fn build_postnorm_encoder(num_blocks: usize) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new(&format!("enc_depth_postnorm_{num_blocks}b"));
    let input = b.add_input("patches", &shape);

    let mut x = input;
    for i in 0..num_blocks {
        let pfx = format!("b{}_", i + 1);

        // Self-attention (no pre-norm)
        let q_w = b.add_input(&format!("{pfx}q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{pfx}k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{pfx}v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let out_w = b.add_input(&format!("{pfx}out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(x, q_w, None, &shape);
        let k = b.add_linear(x, k_w, None, &shape);
        let v = b.add_linear(x, v_w, None, &shape);
        let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let attn_out = b.add_linear(attn, out_w, None, &shape);

        // Post-norm: LayerNorm(x + attn_out)
        let res1 = b.add_binary_add(x, attn_out, &shape);
        let ln1_eps = b.add_input(&format!("{pfx}ln1_eps"), &[1]);
        let ln1_w = b.add_input(&format!("{pfx}ln1_w"), &[HIDDEN_DIM]);
        let ln1_b = b.add_input(&format!("{pfx}ln1_b"), &[HIDDEN_DIM]);
        let mid = b.add_layer_norm(res1, ln1_eps, 1, ln1_w, ln1_b, &shape);

        // FFN
        let ffn_up_w = b.add_input(&format!("{pfx}ffn_up_w"), &[FFN_DIM, HIDDEN_DIM]);
        let ffn_down_w = b.add_input(&format!("{pfx}ffn_down_w"), &[HIDDEN_DIM, FFN_DIM]);
        let ffn_up = b.add_linear(mid, ffn_up_w, None, &ffn_shape);
        let ffn_act = b.add_relu(ffn_up, &ffn_shape);
        let ffn_out = b.add_linear(ffn_act, ffn_down_w, None, &shape);

        // Post-norm: LayerNorm(mid + ffn_out)
        let res2 = b.add_binary_add(mid, ffn_out, &shape);
        let ln2_eps = b.add_input(&format!("{pfx}ln2_eps"), &[1]);
        let ln2_w = b.add_input(&format!("{pfx}ln2_w"), &[HIDDEN_DIM]);
        let ln2_b = b.add_input(&format!("{pfx}ln2_b"), &[HIDDEN_DIM]);
        x = b.add_layer_norm(res2, ln2_eps, 1, ln2_w, ln2_b, &shape);
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid post-norm encoder: {e}"))
}

fn postnorm_encoder_bindings(num_blocks: usize) -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn_up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable]; // patches
    for _ in 0..num_blocks {
        // Attention: Q, K, V, out
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        // Post-attn LN: eps, weight, bias
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        // FFN: up, down
        bindings.push(TensorParamBinding::ConstantTensor(ffn_up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ffn_down_w.clone()));
        // Post-FFN LN: eps, weight, bias
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
    }
    bindings
}

/// Post-norm 2-block encoder IBP: original Transformer / Table Transformer style.
#[test]
fn test_encoder_depth_postnorm_ibp() {
    let def = build_postnorm_encoder(2);
    let bindings = postnorm_encoder_bindings(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through post-norm encoder");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("encoder depth post-norm 2-block IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. Encoder with window attention: local attention depth scaling (IBP)
// ===========================================================================

/// Build a 2-block encoder with window (local) attention pattern.
///
/// Uses `add_multi_head_attention` to simulate window attention blocks
/// (Qwen3-VL pattern). MHA operates on the full (small) sequence which
/// approximates windowed attention at verification scale.
fn build_window_attention_encoder(num_blocks: usize) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let mut b = TensorBlockBuilder::new(&format!("enc_depth_window_{num_blocks}b"));
    let input = b.add_input("patches", &shape);

    let mut x = input;
    for i in 0..num_blocks {
        let pfx = format!("b{}_", i + 1);

        // Pre-norm
        let ln_eps = b.add_input(&format!("{pfx}ln1_eps"), &[1]);
        let ln_w = b.add_input(&format!("{pfx}ln1_w"), &[HIDDEN_DIM]);
        let ln_b = b.add_input(&format!("{pfx}ln1_b"), &[HIDDEN_DIM]);
        let normed = b.add_layer_norm(x, ln_eps, 1, ln_w, ln_b, &shape);

        // Window attention via MHA
        let q_w = b.add_input(&format!("{pfx}q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{pfx}k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{pfx}v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let o_w = b.add_input(&format!("{pfx}o_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let attn_out = b
            .add_multi_head_attention(
                normed,
                q_w,
                k_w,
                v_w,
                o_w,
                NUM_HEADS,
                AttentionMask::Standard,
                &shape,
            )
            .expect("valid window MHA");

        let res1 = b.add_binary_add(x, attn_out, &shape);

        // Pre-FFN norm
        let ln2_eps = b.add_input(&format!("{pfx}ln2_eps"), &[1]);
        let ln2_w = b.add_input(&format!("{pfx}ln2_w"), &[HIDDEN_DIM]);
        let ln2_b = b.add_input(&format!("{pfx}ln2_b"), &[HIDDEN_DIM]);
        let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &shape);

        // FFN
        let ffn_up = b.add_input(&format!("{pfx}ffn_up"), &[FFN_DIM, HIDDEN_DIM]);
        let ffn_down = b.add_input(&format!("{pfx}ffn_down"), &[HIDDEN_DIM, FFN_DIM]);
        let up = b.add_linear(normed2, ffn_up, None, &ffn_shape);
        let act = b.add_gelu(up, &ffn_shape);
        let ffn_out = b.add_linear(act, ffn_down, None, &shape);

        x = b.add_binary_add(res1, ffn_out, &shape);
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid window attention encoder: {e}"))
}

fn window_attention_encoder_bindings(num_blocks: usize) -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn_up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..num_blocks {
        // LN1: eps, weight, bias
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        // MHA: Q, K, V, O
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        // LN2: eps, weight, bias
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        // FFN: up, down
        bindings.push(TensorParamBinding::ConstantTensor(ffn_up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ffn_down_w.clone()));
    }
    bindings
}

/// Window attention 2-block encoder IBP: local attention depth scaling.
#[test]
fn test_encoder_depth_window_attention_ibp() {
    let def = build_window_attention_encoder(2);
    let bindings = window_attention_encoder_bindings(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through window attention encoder");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("encoder depth window-attn 2-block IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. Encoder with cross-attention: depth impact on cross-attn bounds (IBP)
// ===========================================================================

/// Build a 2-block encoder with cross-attention (vision encoder attending to
/// text features, e.g., DETR encoder-decoder or VLM cross-modal attention).
///
/// Block structure: Self-attn -> residual -> Cross-attn(encoder, memory) -> residual -> FFN -> residual.
fn build_cross_attention_encoder() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let mem_shape = [ENC_MEM_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut b = TensorBlockBuilder::new("enc_depth_cross_attn_2b");
    let input = b.add_input("patches", &shape);
    let memory = b.add_input("memory", &mem_shape);

    let mut x = input;
    for i in 0..2 {
        let pfx = format!("b{}_", i + 1);

        // Pre-norm + self-attention
        let ln1_eps = b.add_input(&format!("{pfx}ln1_eps"), &[1]);
        let ln1_w = b.add_input(&format!("{pfx}ln1_w"), &[HIDDEN_DIM]);
        let ln1_b = b.add_input(&format!("{pfx}ln1_b"), &[HIDDEN_DIM]);
        let normed1 = b.add_layer_norm(x, ln1_eps, 1, ln1_w, ln1_b, &shape);

        let sa_q = b.add_input(&format!("{pfx}sa_q"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let sa_k = b.add_input(&format!("{pfx}sa_k"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let sa_v = b.add_input(&format!("{pfx}sa_v"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let sa_o = b.add_input(&format!("{pfx}sa_o"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let q = b.add_linear(normed1, sa_q, None, &shape);
        let k = b.add_linear(normed1, sa_k, None, &shape);
        let v = b.add_linear(normed1, sa_v, None, &shape);
        let sa = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
        let sa_out = b.add_linear(sa, sa_o, None, &shape);
        let res1 = b.add_binary_add(x, sa_out, &shape);

        // Pre-norm + cross-attention (Q from encoder, K/V from memory)
        let ln2_eps = b.add_input(&format!("{pfx}ln2_eps"), &[1]);
        let ln2_w = b.add_input(&format!("{pfx}ln2_w"), &[HIDDEN_DIM]);
        let ln2_b = b.add_input(&format!("{pfx}ln2_b"), &[HIDDEN_DIM]);
        let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &shape);

        let ca_q = b.add_input(&format!("{pfx}ca_q"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let ca_k = b.add_input(&format!("{pfx}ca_k"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let ca_v = b.add_input(&format!("{pfx}ca_v"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let ca_o = b.add_input(&format!("{pfx}ca_o"), &[HIDDEN_DIM, HIDDEN_DIM]);

        let cq = b.add_linear(normed2, ca_q, None, &shape);
        let ck = b.add_linear(memory, ca_k, None, &mem_shape);
        let cv = b.add_linear(memory, ca_v, None, &mem_shape);
        let ca = b.add_attention(cq, ck, cv, AttentionMask::Standard, Some(scale), &shape);
        let ca_out = b.add_linear(ca, ca_o, None, &shape);
        let res2 = b.add_binary_add(res1, ca_out, &shape);

        // Pre-norm + FFN
        let ln3_eps = b.add_input(&format!("{pfx}ln3_eps"), &[1]);
        let ln3_w = b.add_input(&format!("{pfx}ln3_w"), &[HIDDEN_DIM]);
        let ln3_b = b.add_input(&format!("{pfx}ln3_b"), &[HIDDEN_DIM]);
        let normed3 = b.add_layer_norm(res2, ln3_eps, 1, ln3_w, ln3_b, &shape);

        let ffn_up = b.add_input(&format!("{pfx}ffn_up"), &[FFN_DIM, HIDDEN_DIM]);
        let ffn_down = b.add_input(&format!("{pfx}ffn_down"), &[HIDDEN_DIM, FFN_DIM]);
        let up = b.add_linear(normed3, ffn_up, None, &ffn_shape);
        let act = b.add_gelu(up, &ffn_shape);
        let out = b.add_linear(act, ffn_down, None, &shape);

        x = b.add_binary_add(res2, out, &shape);
    }

    b.build(x).expect("valid cross-attention encoder")
}

fn cross_attention_encoder_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let mem = ArrayD::from_elem(IxDyn(&[ENC_MEM_LEN, HIDDEN_DIM]), 0.5f32);
    let ffn_up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![
        TensorParamBinding::Variable,            // patches
        TensorParamBinding::ConstantTensor(mem), // memory
    ];
    for _ in 0..2 {
        // Self-attention: LN(3) + Q/K/V/O(4)
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        // Cross-attention: LN(3) + Q/K/V/O(4)
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
        // FFN: LN(3) + up + down
        bindings.push(TensorParamBinding::ConstantScalar(1e-5));
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ffn_up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ffn_down_w.clone()));
    }
    bindings
}

/// Cross-attention 2-block encoder IBP: depth impact on cross-attn bounds.
#[test]
fn test_encoder_depth_cross_attention_ibp() {
    let def = build_cross_attention_encoder();
    let bindings = cross_attention_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-attention encoder");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("encoder depth cross-attn 2-block IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. SigLIP2 encoder depth: Granite-Docling ViT depth profile (IBP)
// ===========================================================================

/// SigLIP2 encoder uses pre-norm LayerNorm + MHA + GELU FFN blocks.
/// Granite-Docling uses 27 encoder blocks; we test 4-block depth profile.
#[test]
fn test_encoder_depth_siglip2_profile_ibp() {
    // SigLIP2 / Granite-Docling: pre-norm LayerNorm, 4-block depth
    let def = build_n_block_encoder(4);
    let bindings = n_block_encoder_bindings(4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through SigLIP2-style encoder");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("SigLIP2 4-block encoder IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 11. Qwen3-VL encoder depth: window ViT depth profile (IBP)
// ===========================================================================

/// Qwen3-VL uses window attention + RMSNorm + SwiGLU FFN blocks.
/// Production uses 32 blocks; we test 4-block depth profile.
#[test]
fn test_encoder_depth_qwen3vl_profile_ibp() {
    // Qwen3-VL: pre-norm RMSNorm + SwiGLU, 4-block depth
    let def = build_prenorm_rmsnorm_encoder(4);
    let bindings = prenorm_rmsnorm_encoder_bindings(4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through Qwen3-VL-style encoder");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("Qwen3-VL 4-block encoder IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 12. SVTR encoder depth: PaddleOCR recognition encoder (IBP)
// ===========================================================================

/// SVTR uses post-norm LayerNorm + MHA + MLP(GELU) blocks.
/// PaddleOCR SVTR uses 8 encoder blocks; we test 4-block depth profile.
#[test]
fn test_encoder_depth_svtr_profile_ibp() {
    // SVTR: post-norm pattern, 4-block depth
    let def = build_postnorm_encoder(4);
    let bindings = postnorm_encoder_bindings(4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through SVTR-style encoder");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("SVTR 4-block encoder IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 13. Depth vs CROWN tightness: CROWN advantage at increasing depth (CROWN)
// ===========================================================================

/// Compare CROWN vs IBP bound widths at 1-block and 2-block depths.
///
/// CROWN linearization should (when not falling back) produce tighter
/// bounds than IBP. This test logs the width at each depth for the
/// tightness advantage curve.
#[test]
fn test_encoder_depth_crown_tightness_vs_depth() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    for &depth in &[1usize, 2] {
        let def = build_n_block_encoder(depth);
        let bindings = n_block_encoder_bindings(depth);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        // IBP baseline
        let ibp_output = graph.propagate_ibp(&input).expect("IBP");
        let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
        let ibp_width = ibp_hi - ibp_lo;

        // CROWN
        let (method, crown_output, fallback_reason) =
            assert_crown_tighter_when_not_fallback(&graph, &input);
        let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
        let crown_width = crown_hi - crown_lo;

        eprintln!(
            "encoder depth={depth}: IBP width={ibp_width:.4}, CROWN width={crown_width:.4}, \
             method={method:?}, bounds=[{crown_lo}, {crown_hi}]"
        );
        if let Some(reason) = &fallback_reason {
            eprintln!("  Fallback reason: {reason}");
        }

        assert!(ibp_width.is_finite(), "IBP width finite at depth {depth}");
        assert!(
            crown_width.is_finite(),
            "CROWN width finite at depth {depth}"
        );
    }
}

// ===========================================================================
// 14. Encoder depth monotone: deeper -> wider bounds (verified property) (IBP)
// ===========================================================================

/// Verify that encoder depth monotonically widens IBP bounds.
///
/// Tests depths 1, 2, 4, 8 and asserts each successive depth produces
/// bounds at least as wide as the previous. This is a fundamental property
/// of IBP over-approximation accumulation through depth.
#[test]
fn test_encoder_depth_monotone_widening_property() {
    let depths = [1usize, 2, 4, 8];
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let mut widths = Vec::new();

    for &depth in &depths {
        let def = build_n_block_encoder(depth);
        let bindings = n_block_encoder_bindings(depth);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        let output = graph.propagate_ibp(&input).expect("IBP");

        assert_bounds_valid(&output);
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("encoder depth monotone: depth={depth}, width={width:.6}");

        widths.push((depth, width));
    }

    // Verify monotone widening (with small tolerance for numerical noise).
    for i in 1..widths.len() {
        let (d_prev, w_prev) = widths[i - 1];
        let (d_curr, w_curr) = widths[i];
        let tolerance = w_prev * 0.01 + 1e-4;
        assert!(
            w_curr >= w_prev - tolerance,
            "depth {d_curr} width {w_curr:.6} should be >= depth {d_prev} width {w_prev:.6}"
        );
    }
}

// ===========================================================================
// 15. Encoder depth + head: full encoder -> projection -> sigmoid (IBP)
// ===========================================================================

/// Build a 2-block encoder followed by a classification head:
/// encoder -> Linear(HIDDEN_DIM, PROJ_DIM) -> sigmoid.
///
/// This tests the full encoder-to-output pipeline with depth.
fn build_encoder_with_head() -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let out_shape = [SEQ_LEN, PROJ_DIM];

    let mut b = TensorBlockBuilder::new("enc_depth_with_head");
    let input = b.add_input("patches", &shape);

    // 2-block encoder
    let mut x = input;
    for i in 0..2 {
        x = add_encoder_block_layernorm(&mut b, x, &format!("b{}_", i + 1));
    }

    // Final LayerNorm
    let final_eps = b.add_input("final_ln_eps", &[1]);
    let final_w = b.add_input("final_ln_w", &[HIDDEN_DIM]);
    let final_b = b.add_input("final_ln_b", &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(x, final_eps, 1, final_w, final_b, &shape);

    // Projection head: Linear -> sigmoid
    let proj_w = b.add_input("proj_w", &[PROJ_DIM, HIDDEN_DIM]);
    let proj_b = b.add_input("proj_b", &[PROJ_DIM]);
    let projected = b.add_linear(normed, proj_w, Some(proj_b), &out_shape);
    let out = b.add_sigmoid(projected, &out_shape);

    b.build(out).expect("valid encoder + head kernel")
}

fn encoder_with_head_bindings() -> Vec<TensorParamBinding> {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    let mut bindings = vec![TensorParamBinding::Variable]; // patches

    // 2 encoder blocks
    for _ in 0..2 {
        push_encoder_block_layernorm_bindings(&mut bindings);
    }

    // Final LayerNorm
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ln_w));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b));

    // Projection head
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[PROJ_DIM, HIDDEN_DIM]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[PROJ_DIM]),
        0.0f32,
    )));

    bindings
}

/// Full encoder -> projection -> sigmoid: output bounded in (0, 1).
#[test]
fn test_encoder_depth_with_head_ibp() {
    let def = build_encoder_with_head();
    let bindings = encoder_with_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder + head");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, PROJ_DIM],
        "encoder + head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("encoder depth + sigmoid head IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid output: bounds must be in [0, 1]
    assert!(lo_min >= -1e-6, "sigmoid output lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-6,
        "sigmoid output upper <= 1, got {hi_max}"
    );
}
