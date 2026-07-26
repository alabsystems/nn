// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: transformer block composite → NY verification.
//!
//! Validates that `add_transformer_block()` decomposes into
//! LayerNorm → MHA → BinaryAdd(residual) → LayerNorm → Linear → GELU → Linear
//! → BinaryAdd(residual) and that the resulting `GraphNetwork` propagates
//! bounds via IBP and CROWN.
//!
//! Part of #811.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, uniform_bounds, verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::{AttentionMask, TransformerBlockConfig, TransformerBlockWeights};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build a single transformer block kernel: input [T, D], 2-head attention.
///
/// Input is the only Variable; all weights are ConstantTensor.
/// Uses small constant weights (0.02) for numerical stability.
fn build_transformer_kernel(
    name: &str,
    seq_len: usize,
    model_dim: usize,
    num_heads: usize,
    ffn_hidden_dim: usize,
) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);

    let input = b.add_input("x", &[seq_len, model_dim]);
    let eps = b.add_input("eps", &[1]);
    let ln1_w = b.add_input("ln1_weight", &[model_dim]);
    let ln1_b = b.add_input("ln1_bias", &[model_dim]);
    let ln2_w = b.add_input("ln2_weight", &[model_dim]);
    let ln2_b = b.add_input("ln2_bias", &[model_dim]);
    let q_w = b.add_input("q_weight", &[model_dim, model_dim]);
    let k_w = b.add_input("k_weight", &[model_dim, model_dim]);
    let v_w = b.add_input("v_weight", &[model_dim, model_dim]);
    let out_w = b.add_input("out_weight", &[model_dim, model_dim]);
    let ffn1_w = b.add_input("ffn1_weight", &[ffn_hidden_dim, model_dim]);
    let ffn2_w = b.add_input("ffn2_weight", &[model_dim, ffn_hidden_dim]);

    let weights = TransformerBlockWeights {
        ln1_weight: ln1_w,
        ln1_bias: ln1_b,
        ln2_weight: ln2_w,
        ln2_bias: ln2_b,
        q_weight: q_w,
        k_weight: k_w,
        v_weight: v_w,
        out_weight: out_w,
        ffn1_weight: ffn1_w,
        ffn2_weight: ffn2_w,
        eps,
    };

    let config = TransformerBlockConfig {
        num_heads,
        mask: AttentionMask::Standard,
        ffn_hidden_dim,
    };

    let out = b
        .add_transformer_block(input, &weights, &config)
        .expect("valid transformer block");
    b.build(out).expect("valid kernel")
}

/// Bindings for a single transformer block.
///
/// Input is Variable, all 11 weight/bias/eps parameters are ConstantTensor.
fn transformer_bindings(model_dim: usize, ffn_hidden_dim: usize) -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[model_dim, model_dim]), 0.02f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[ffn_hidden_dim, model_dim]), 0.02f32);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[model_dim, ffn_hidden_dim]), 0.02f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[model_dim]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[model_dim]), 0.0f32);

    vec![
        TensorParamBinding::Variable,                       // input [T, D]
        TensorParamBinding::ConstantScalar(1e-5),           // eps [1]
        TensorParamBinding::ConstantTensor(ln_w.clone()),   // ln1_weight [D]
        TensorParamBinding::ConstantTensor(ln_b.clone()),   // ln1_bias [D]
        TensorParamBinding::ConstantTensor(ln_w),           // ln2_weight [D]
        TensorParamBinding::ConstantTensor(ln_b),           // ln2_bias [D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // q_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // k_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj.clone()), // v_weight [D, D]
        TensorParamBinding::ConstantTensor(w_proj),         // out_weight [D, D]
        TensorParamBinding::ConstantTensor(w_ffn1),         // ffn1_weight [ffn, D]
        TensorParamBinding::ConstantTensor(w_ffn2),         // ffn2_weight [D, ffn]
    ]
}

// ---------------------------------------------------------------------------
// AC4: Graph construction — single-block, 2-head attention
// ---------------------------------------------------------------------------

/// Transformer block translates into a valid NY GraphNetwork.
#[test]
fn test_transformer_block_graph_builds() {
    let (t, d, h, ffn) = (4, 8, 2, 16);
    let def = build_transformer_kernel("tf_build", t, d, h, ffn);

    let bindings = transformer_bindings(d, ffn);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("transformer graph must build");
    assert!(
        graph.num_nodes() >= 10,
        "transformer block needs many translation nodes, got {}",
        graph.num_nodes()
    );
}

// ---------------------------------------------------------------------------
// AC4: IBP propagation — single-block
// ---------------------------------------------------------------------------

/// IBP bounds propagate through a single transformer block.
///
/// Single variable input [T, D] in [-1, 1]. All weights are small constants.
/// Output should have finite, valid bounds with shape [T, D].
#[test]
fn test_transformer_block_ibp_propagates() {
    let (t, d, h, ffn) = (4, 8, 2, 16);
    let def = build_transformer_kernel("tf_ibp", t, d, h, ffn);
    let bindings = transformer_bindings(d, ffn);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("transformer graph");

    let input = uniform_bounds(&[t, d], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through transformer block");
    let (lo, _hi) = output.lower_upper();

    // Single variable: output shape matches kernel output [T, D].
    assert_eq!(lo.shape(), &[t, d], "output shape [T, D]");

    assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// AC4: CROWN propagation — single-block
// ---------------------------------------------------------------------------

/// CROWN bounds propagate through a single transformer block.
/// When CROWN succeeds (no fallback), verifies tighter-than-IBP invariant.
#[test]
fn test_transformer_block_crown_propagates() {
    let (t, d, h, ffn) = (4, 8, 2, 16);
    let def = build_transformer_kernel("tf_crown", t, d, h, ffn);
    let bindings = transformer_bindings(d, ffn);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("transformer graph");

    let input = uniform_bounds(&[t, d], 1.0);
    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (lo, _hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[t, d], "output shape [T, D]");

    eprintln!("Transformer block: method={method:?}");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
}

// ---------------------------------------------------------------------------
// AC5: Multi-block composition — 2-block stack
// ---------------------------------------------------------------------------

/// Build a 2-block transformer stack: block1 → block2.
fn build_two_block_kernel(
    name: &str,
    seq_len: usize,
    model_dim: usize,
    num_heads: usize,
    ffn_hidden_dim: usize,
) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);

    let input = b.add_input("x", &[seq_len, model_dim]);
    let eps = b.add_input("eps", &[1]);

    // Block 1 weights
    let ln1_w1 = b.add_input("b1_ln1_w", &[model_dim]);
    let ln1_b1 = b.add_input("b1_ln1_b", &[model_dim]);
    let ln2_w1 = b.add_input("b1_ln2_w", &[model_dim]);
    let ln2_b1 = b.add_input("b1_ln2_b", &[model_dim]);
    let q_w1 = b.add_input("b1_q_w", &[model_dim, model_dim]);
    let k_w1 = b.add_input("b1_k_w", &[model_dim, model_dim]);
    let v_w1 = b.add_input("b1_v_w", &[model_dim, model_dim]);
    let out_w1 = b.add_input("b1_out_w", &[model_dim, model_dim]);
    let ffn1_w1 = b.add_input("b1_ffn1_w", &[ffn_hidden_dim, model_dim]);
    let ffn2_w1 = b.add_input("b1_ffn2_w", &[model_dim, ffn_hidden_dim]);

    // Block 2 weights
    let ln1_w2 = b.add_input("b2_ln1_w", &[model_dim]);
    let ln1_b2 = b.add_input("b2_ln1_b", &[model_dim]);
    let ln2_w2 = b.add_input("b2_ln2_w", &[model_dim]);
    let ln2_b2 = b.add_input("b2_ln2_b", &[model_dim]);
    let q_w2 = b.add_input("b2_q_w", &[model_dim, model_dim]);
    let k_w2 = b.add_input("b2_k_w", &[model_dim, model_dim]);
    let v_w2 = b.add_input("b2_v_w", &[model_dim, model_dim]);
    let out_w2 = b.add_input("b2_out_w", &[model_dim, model_dim]);
    let ffn1_w2 = b.add_input("b2_ffn1_w", &[ffn_hidden_dim, model_dim]);
    let ffn2_w2 = b.add_input("b2_ffn2_w", &[model_dim, ffn_hidden_dim]);

    let config = TransformerBlockConfig {
        num_heads,
        mask: AttentionMask::Standard,
        ffn_hidden_dim,
    };

    let weights1 = TransformerBlockWeights {
        ln1_weight: ln1_w1,
        ln1_bias: ln1_b1,
        ln2_weight: ln2_w1,
        ln2_bias: ln2_b1,
        q_weight: q_w1,
        k_weight: k_w1,
        v_weight: v_w1,
        out_weight: out_w1,
        ffn1_weight: ffn1_w1,
        ffn2_weight: ffn2_w1,
        eps,
    };

    let block1 = b
        .add_transformer_block(input, &weights1, &config)
        .expect("block 1");

    let weights2 = TransformerBlockWeights {
        ln1_weight: ln1_w2,
        ln1_bias: ln1_b2,
        ln2_weight: ln2_w2,
        ln2_bias: ln2_b2,
        q_weight: q_w2,
        k_weight: k_w2,
        v_weight: v_w2,
        out_weight: out_w2,
        ffn1_weight: ffn1_w2,
        ffn2_weight: ffn2_w2,
        eps, // shared eps
    };

    let block2 = b
        .add_transformer_block(block1, &weights2, &config)
        .expect("block 2");

    b.build(block2).expect("valid two-block kernel")
}

/// Bindings for a 2-block transformer stack.
///
/// Input is Variable, eps is shared, all 20 weight/bias parameters are ConstantTensor.
fn two_block_bindings(model_dim: usize, ffn_hidden_dim: usize) -> Vec<TensorParamBinding> {
    let w_proj = ArrayD::from_elem(IxDyn(&[model_dim, model_dim]), 0.02f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[ffn_hidden_dim, model_dim]), 0.02f32);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[model_dim, ffn_hidden_dim]), 0.02f32);
    let ln_w = ArrayD::from_elem(IxDyn(&[model_dim]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[model_dim]), 0.0f32);

    vec![
        TensorParamBinding::Variable,             // input
        TensorParamBinding::ConstantScalar(1e-5), // eps (shared)
        // Block 1 weights
        TensorParamBinding::ConstantTensor(ln_w.clone()), // b1_ln1_w
        TensorParamBinding::ConstantTensor(ln_b.clone()), // b1_ln1_b
        TensorParamBinding::ConstantTensor(ln_w.clone()), // b1_ln2_w
        TensorParamBinding::ConstantTensor(ln_b.clone()), // b1_ln2_b
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b1_q_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b1_k_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b1_v_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b1_out_w
        TensorParamBinding::ConstantTensor(w_ffn1.clone()), // b1_ffn1_w
        TensorParamBinding::ConstantTensor(w_ffn2.clone()), // b1_ffn2_w
        // Block 2 weights
        TensorParamBinding::ConstantTensor(ln_w.clone()), // b2_ln1_w
        TensorParamBinding::ConstantTensor(ln_b.clone()), // b2_ln1_b
        TensorParamBinding::ConstantTensor(ln_w),         // b2_ln2_w
        TensorParamBinding::ConstantTensor(ln_b),         // b2_ln2_b
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b2_q_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b2_k_w
        TensorParamBinding::ConstantTensor(w_proj.clone()), // b2_v_w
        TensorParamBinding::ConstantTensor(w_proj),       // b2_out_w
        TensorParamBinding::ConstantTensor(w_ffn1),       // b2_ffn1_w
        TensorParamBinding::ConstantTensor(w_ffn2),       // b2_ffn2_w
    ]
}

/// 2-block transformer stack translates into a valid GraphNetwork (AC5).
#[test]
fn test_two_block_transformer_graph_builds() {
    let (t, d, h, ffn) = (4, 8, 2, 16);
    let def = build_two_block_kernel("tf_2block", t, d, h, ffn);

    let bindings = two_block_bindings(d, ffn);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("2-block transformer graph");
    assert!(
        graph.num_nodes() >= 20,
        "2-block transformer needs many nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through a 2-block transformer stack (AC5).
#[test]
fn test_two_block_transformer_ibp_propagates() {
    let (t, d, h, ffn) = (4, 8, 2, 16);
    let def = build_two_block_kernel("tf_2block_ibp", t, d, h, ffn);
    let bindings = two_block_bindings(d, ffn);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("2-block graph");

    let input = uniform_bounds(&[t, d], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 2-block transformer");
    let (lo, _hi) = output.lower_upper();

    assert_eq!(lo.shape(), &[t, d], "output shape [T, D]");

    assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// AC7: verify_tensor_and_record integration
// ---------------------------------------------------------------------------

/// Full pipeline: translate + propagate + record under "transformer_block" key.
#[test]
fn test_transformer_block_verify_and_record() {
    let (t, d, h, ffn) = (4, 8, 2, 16);
    let def = build_transformer_kernel("tf_pipeline", t, d, h, ffn);
    let bindings = transformer_bindings(d, ffn);
    let input = uniform_bounds(&[t, d], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "transformer_block");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[t, d], "output shape");
}
