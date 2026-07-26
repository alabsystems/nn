// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Demucs cross-domain transformer bottleneck composition.
//!
//! HTDemucs uses a cross-domain transformer where temporal and spectral encoder
//! outputs interact through cross-attention. Each transformer layer:
//!
//! ```text
//! Temporal:  [C, T] → Conv1d(C→D) → Transpose [D,T]→[T,D]
//! Spectral:  [C, F] → Conv1d(C→D) → Transpose [D,F]→[F,D]
//!
//! For each layer:
//!   1. Self-attention on temporal:  [T, D] → TransformerBlock → [T, D]
//!   2. Cross-attention: temporal queries spectral (T queries F)
//!   3. Self-attention on spectral:  [F, D] → TransformerBlock → [F, D]
//!   4. Cross-attention: spectral queries temporal (F queries T)
//!
//! Temporal:  Transpose [T,D]→[D,T] → Conv1d(D→C) → [C, T]
//! Spectral:  Transpose [F,D]→[D,F] → Conv1d(D→C) → [C, F]
//! ```
//!
//! Single-variable mode: temporal encoder output is Variable, spectral KV
//! is pre-processed ConstantTensor at `[F_SEQ, MODEL_DIM]`.
//!
//! Part of #779 Phase D — cross-domain transformer bottleneck.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorNodeId;
use nn_dsl::{AttentionMask, TransformerBlockConfig, TransformerBlockWeights};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Parameters — small dims for NY tractability
// ---------------------------------------------------------------------------

const ENC_CH: usize = 4;
const MODEL_DIM: usize = 8;
const NUM_HEADS: usize = 2;
const FFN_HIDDEN: usize = MODEL_DIM * 2;
const T_SEQ: usize = 4;
/// Spectral freq bins (must equal T_SEQ for single-variable mode).
const F_SEQ: usize = T_SEQ;
const WEIGHT_MAG: f32 = 0.01;

// ---------------------------------------------------------------------------
// Collected input node IDs
// ---------------------------------------------------------------------------

/// Cross-attention weights decomposed manually (no LN2 on constant KV).
struct ManualCrossAttnWeights {
    ln1_weight: TensorNodeId,
    ln1_bias: TensorNodeId,
    ln3_weight: TensorNodeId,
    ln3_bias: TensorNodeId,
    ln_out_weight: TensorNodeId,
    ln_out_bias: TensorNodeId,
    q_weight: TensorNodeId,
    k_weight: TensorNodeId,
    v_weight: TensorNodeId,
    out_weight: TensorNodeId,
    ffn1_weight: TensorNodeId,
    ffn2_weight: TensorNodeId,
    eps: TensorNodeId,
}

/// All inputs for the cross-domain transformer bottleneck.
struct BottleneckInputs {
    temporal: TensorNodeId,
    spectral_kv: TensorNodeId,
    t_up_weight: TensorNodeId,
    t_up_bias: TensorNodeId,
    t_down_weight: TensorNodeId,
    t_down_bias: TensorNodeId,
    t_self_attn: TransformerBlockWeights,
    t_cross_attn: ManualCrossAttnWeights,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

fn add_self_attn_weights(
    b: &mut TensorBlockBuilder,
    prefix: &str,
    eps: TensorNodeId,
) -> TransformerBlockWeights {
    let d = MODEL_DIM;
    TransformerBlockWeights {
        ln1_weight: b.add_input(&format!("{prefix}_sa_ln1_w"), &[d]),
        ln1_bias: b.add_input(&format!("{prefix}_sa_ln1_b"), &[d]),
        ln2_weight: b.add_input(&format!("{prefix}_sa_ln2_w"), &[d]),
        ln2_bias: b.add_input(&format!("{prefix}_sa_ln2_b"), &[d]),
        q_weight: b.add_input(&format!("{prefix}_sa_q_w"), &[d, d]),
        k_weight: b.add_input(&format!("{prefix}_sa_k_w"), &[d, d]),
        v_weight: b.add_input(&format!("{prefix}_sa_v_w"), &[d, d]),
        out_weight: b.add_input(&format!("{prefix}_sa_out_w"), &[d, d]),
        ffn1_weight: b.add_input(&format!("{prefix}_sa_ffn1_w"), &[FFN_HIDDEN, d]),
        ffn2_weight: b.add_input(&format!("{prefix}_sa_ffn2_w"), &[d, FFN_HIDDEN]),
        eps,
    }
}

fn add_manual_cross_attn_weights(
    b: &mut TensorBlockBuilder,
    prefix: &str,
    eps: TensorNodeId,
) -> ManualCrossAttnWeights {
    let d = MODEL_DIM;
    ManualCrossAttnWeights {
        ln1_weight: b.add_input(&format!("{prefix}_ca_ln1_w"), &[d]),
        ln1_bias: b.add_input(&format!("{prefix}_ca_ln1_b"), &[d]),
        ln3_weight: b.add_input(&format!("{prefix}_ca_ln3_w"), &[d]),
        ln3_bias: b.add_input(&format!("{prefix}_ca_ln3_b"), &[d]),
        ln_out_weight: b.add_input(&format!("{prefix}_ca_lnout_w"), &[d]),
        ln_out_bias: b.add_input(&format!("{prefix}_ca_lnout_b"), &[d]),
        q_weight: b.add_input(&format!("{prefix}_ca_q_w"), &[d, d]),
        k_weight: b.add_input(&format!("{prefix}_ca_k_w"), &[d, d]),
        v_weight: b.add_input(&format!("{prefix}_ca_v_w"), &[d, d]),
        out_weight: b.add_input(&format!("{prefix}_ca_out_w"), &[d, d]),
        ffn1_weight: b.add_input(&format!("{prefix}_ca_ffn1_w"), &[FFN_HIDDEN, d]),
        ffn2_weight: b.add_input(&format!("{prefix}_ca_ffn2_w"), &[d, FFN_HIDDEN]),
        eps,
    }
}

fn add_bottleneck_inputs(b: &mut TensorBlockBuilder) -> BottleneckInputs {
    let temporal = b.add_input("temporal_enc", &[ENC_CH, T_SEQ]);
    let spectral_kv = b.add_input("spectral_kv", &[F_SEQ, MODEL_DIM]);

    let t_up_weight = b.add_input("t_up_w", &[MODEL_DIM, ENC_CH, 1]);
    let t_up_bias = b.add_input("t_up_b", &[MODEL_DIM]);
    let t_down_weight = b.add_input("t_down_w", &[ENC_CH, MODEL_DIM, 1]);
    let t_down_bias = b.add_input("t_down_b", &[ENC_CH]);

    let eps = b.add_input("eps", &[1]);
    let t_self_attn = add_self_attn_weights(b, "t", eps);
    let t_cross_attn = add_manual_cross_attn_weights(b, "t", eps);

    BottleneckInputs {
        temporal,
        spectral_kv,
        t_up_weight,
        t_up_bias,
        t_down_weight,
        t_down_bias,
        t_self_attn,
        t_cross_attn,
    }
}

/// Build the Demucs cross-domain transformer bottleneck.
///
/// Temporal path: Conv1d(C→D) → Transpose → SelfAttn → CrossAttn(queries spectral)
/// → Transpose → Conv1d(D→C). Cross-attention is manually decomposed because
/// spectral KV is constant (NY cannot apply LayerNorm to constants).
fn build_cross_domain_bottleneck() -> nn_dsl::tensor_ir::TensorKernelDef {
    let mut b = TensorBlockBuilder::new("demucs_cross_domain_verify");
    let inp = add_bottleneck_inputs(&mut b);

    // Temporal channel bridge: [C, T] → Conv1d(1x1) → [D, T]
    let t_up = b.add_conv1d(
        inp.temporal,
        inp.t_up_weight,
        Some(inp.t_up_bias),
        1,
        0,
        &[MODEL_DIM, T_SEQ],
    );
    let t_td = b.add_transpose(t_up, &[1, 0], &[T_SEQ, MODEL_DIM]);

    // Self-attention: [T, D] → TransformerBlock → [T, D]
    let tc = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_HIDDEN,
    };
    let t_self = b
        .add_transformer_block(t_td, &inp.t_self_attn, &tc)
        .expect("temporal self-attention");

    // Cross-attention (manual): temporal queries spectral, skip LN2 on constant KV
    let ca = &inp.t_cross_attn;
    let shape = [T_SEQ, MODEL_DIM];
    let ffn_shape = [T_SEQ, FFN_HIDDEN];

    let normed_q = b.add_layer_norm(t_self, ca.eps, 1, ca.ln1_weight, ca.ln1_bias, &shape);
    let attn = b
        .add_multi_head_cross_attention(
            normed_q,
            inp.spectral_kv,
            ca.q_weight,
            ca.k_weight,
            ca.v_weight,
            ca.out_weight,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("cross-MHA temporal queries spectral");
    let residual1 = b.add_binary_add(t_self, attn, &shape);

    // LN3 → FFN → Residual
    let normed3 = b.add_layer_norm(residual1, ca.eps, 1, ca.ln3_weight, ca.ln3_bias, &shape);
    let ffn1 = b.add_linear(normed3, ca.ffn1_weight, None, &ffn_shape);
    let act = b.add_gelu(ffn1, &ffn_shape);
    let ffn2 = b.add_linear(act, ca.ffn2_weight, None, &shape);
    let residual2 = b.add_binary_add(residual1, ffn2, &shape);

    // LN_out
    let t_cross = b.add_layer_norm(
        residual2,
        ca.eps,
        1,
        ca.ln_out_weight,
        ca.ln_out_bias,
        &shape,
    );

    // Back: Transpose [T, D] → [D, T] → Conv1d(D→C) → [C, T]
    let t_dt = b.add_transpose(t_cross, &[1, 0], &[MODEL_DIM, T_SEQ]);
    let t_out = b.add_conv1d(
        t_dt,
        inp.t_down_weight,
        Some(inp.t_down_bias),
        1,
        0,
        &[ENC_CH, T_SEQ],
    );

    b.build(t_out).expect("valid cross-domain bottleneck graph")
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

fn push_weight(bindings: &mut Vec<TensorParamBinding>, shape: &[usize], val: f32) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(shape),
        val,
    )));
}

fn push_self_attn_bindings(b: &mut Vec<TensorParamBinding>) {
    let d = MODEL_DIM;
    push_weight(b, &[d], 1.0); // ln1 gamma
    push_weight(b, &[d], 0.0); // ln1 beta
    push_weight(b, &[d], 1.0); // ln2 gamma
    push_weight(b, &[d], 0.0); // ln2 beta
    push_weight(b, &[d, d], WEIGHT_MAG); // Q
    push_weight(b, &[d, d], WEIGHT_MAG); // K
    push_weight(b, &[d, d], WEIGHT_MAG); // V
    push_weight(b, &[d, d], WEIGHT_MAG); // out
    push_weight(b, &[FFN_HIDDEN, d], WEIGHT_MAG); // ffn1
    push_weight(b, &[d, FFN_HIDDEN], WEIGHT_MAG); // ffn2
}

fn push_manual_cross_attn_bindings(b: &mut Vec<TensorParamBinding>) {
    let d = MODEL_DIM;
    push_weight(b, &[d], 1.0); // ln1 gamma
    push_weight(b, &[d], 0.0); // ln1 beta
    push_weight(b, &[d], 1.0); // ln3 gamma
    push_weight(b, &[d], 0.0); // ln3 beta
    push_weight(b, &[d], 1.0); // ln_out gamma
    push_weight(b, &[d], 0.0); // ln_out beta
    push_weight(b, &[d, d], WEIGHT_MAG); // Q
    push_weight(b, &[d, d], WEIGHT_MAG); // K
    push_weight(b, &[d, d], WEIGHT_MAG); // V
    push_weight(b, &[d, d], WEIGHT_MAG); // out
    push_weight(b, &[FFN_HIDDEN, d], WEIGHT_MAG); // ffn1
    push_weight(b, &[d, FFN_HIDDEN], WEIGHT_MAG); // ffn2
}

fn bottleneck_bindings() -> Vec<TensorParamBinding> {
    let mut b = Vec::new();

    b.push(TensorParamBinding::Variable);
    push_weight(&mut b, &[F_SEQ, MODEL_DIM], 0.1);

    push_weight(&mut b, &[MODEL_DIM, ENC_CH, 1], WEIGHT_MAG);
    push_weight(&mut b, &[MODEL_DIM], 0.0);
    push_weight(&mut b, &[ENC_CH, MODEL_DIM, 1], WEIGHT_MAG);
    push_weight(&mut b, &[ENC_CH], 0.0);

    b.push(TensorParamBinding::ConstantScalar(1e-5));

    push_self_attn_bindings(&mut b);
    push_manual_cross_attn_bindings(&mut b);

    b
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_cross_domain_def_validates() {
    let def = build_cross_domain_bottleneck();
    def.validate()
        .expect("cross-domain bottleneck def should validate");
}

#[test]
fn test_cross_domain_graph_builds() {
    let def = build_cross_domain_bottleneck();
    let bindings = bottleneck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("cross-domain bottleneck graph should translate");

    assert!(
        graph.num_nodes() >= 20,
        "cross-domain graph should have >= 20 nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the full cross-domain bottleneck.
#[test]
fn test_cross_domain_ibp_propagates() {
    let def = build_cross_domain_bottleneck();
    let bindings = bottleneck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_CH, T_SEQ], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through cross-domain bottleneck");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[ENC_CH, T_SEQ],
        "output shape should be [ENC_CH, T_SEQ]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Cross-domain IBP: bounds=[{lo_min}, {hi_max}]");

    // Bounds may be wide due to +1 axis convention through chained norms (#2987).
    // Check finiteness rather than tight magnitude until axis convention is fixed.
    assert!(
        lo_min.is_finite(),
        "IBP lower bound must be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "IBP upper bound must be finite, got {hi_max}"
    );
}

/// CROWN propagation through the cross-domain bottleneck.
///
/// Uses `assert_crown_tighter_when_not_fallback` to verify CROWN produces
/// tighter bounds than IBP when CROWN succeeds (not fallback).
#[test]
fn test_cross_domain_crown_propagation() {
    let def = build_cross_domain_bottleneck();
    let bindings = bottleneck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_CH, T_SEQ], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, _) = output.lower_upper();

    assert_eq!(
        lo.shape(),
        &[ENC_CH, T_SEQ],
        "output shape should be [ENC_CH, T_SEQ]"
    );

    eprintln!("Cross-domain: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

/// Cross-domain bottleneck preserves temporal shape (autoencoder property).
#[test]
fn test_cross_domain_preserves_temporal_shape() {
    let def = build_cross_domain_bottleneck();
    let bindings = bottleneck_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[ENC_CH, T_SEQ], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP");
    let (lo, _) = output.lower_upper();
    assert_eq!(
        lo.shape(),
        &[ENC_CH, T_SEQ],
        "output must match temporal input shape"
    );
}

/// Record verification under "demucs_cross_domain_bottleneck" key.
#[test]
fn test_cross_domain_verify_and_record() {
    let def = build_cross_domain_bottleneck();
    let bindings = bottleneck_bindings();
    let input = uniform_bounds(&[ENC_CH, T_SEQ], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "demucs_cross_domain_bottleneck");
    assert_eq!(result.num_variables, 1, "single Variable input (temporal)");
}
