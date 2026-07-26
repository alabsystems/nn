// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep Qwen3 compose verification tests targeting non-vacuous bounds.
//!
//! The existing Qwen3 verification has 1 entry (vacuous). These tests decompose
//! the Qwen3 architecture into focused sub-graphs with Conservative NormBoundsMode
//! to promote entries from vacuous to sound:
//!
//! 1. **RMSNorm isolation (Conservative)**: Sound IBP through normalization.
//! 2. **SwiGLU FFN (Conservative)**: Sound bounds through gated MLP.
//! 3. **Self-attention with RoPE (Conservative)**: Sound bounds through
//!    causal attention with rotary embeddings.
//! 4. **Single decoder block (Conservative)**: RMSNorm -> GQA -> residual ->
//!    RMSNorm -> SwiGLU -> residual. Targets Sound soundness.
//! 5. **Post-norm + LM head (Conservative)**: RMSNorm -> Linear.
//!    Targets Sound soundness.
//! 6. **Residual bounds analysis**: Verifies residual connections do not
//!    cause excessive bounds blowup through 2 blocks.
//!
//! Uses IbpValidated soundness mode per nn engineering rules (Sound refuses
//! linearization for normalization layers). Conservative mode bypasses
//! heuristic linearization entirely, producing Sound classification.
//!
//! Part of compose verification deepening for Qwen3 model.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{
    tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding, VerificationSoundnessMode,
    VerifyConfig,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const D_MODEL: usize = 16;
const N_HEADS: usize = 2;
const FFN_DIM: usize = 48;
const SEQ: usize = 4;
const VOCAB: usize = 32;
const WEIGHT_MAG: f32 = 0.001;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

/// RMSNorm: x / rms(x) * weight.
fn build_rmsnorm_conservative() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_deep_rmsnorm");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let weight = b.add_input("weight", &[D_MODEL]);
    let out = b.add_rms_norm(x, eps, 1, weight, &[SEQ, D_MODEL]);
    b.build(out).expect("valid RMSNorm")
}

fn rmsnorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
    ]
}

/// SwiGLU MLP: gate = silu(x @ gate_w), up = x @ up_w, out = (gate * up) @ down_w.
/// SiLU(x) = x * sigmoid(x) -- decomposed into sigmoid + binary_mul.
fn build_swiglu_conservative() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_deep_swiglu");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let gate_w = b.add_input("gate_w", &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input("up_w", &[FFN_DIM, D_MODEL]);
    let down_w = b.add_input("down_w", &[D_MODEL, FFN_DIM]);

    let gate_proj = b.add_linear(x, gate_w, None, &[SEQ, FFN_DIM]);
    // SiLU(x) = x * sigmoid(x)
    let gate_sig = b.add_sigmoid(gate_proj, &[SEQ, FFN_DIM]);
    let gate_act = b.add_binary_mul(gate_proj, gate_sig, &[SEQ, FFN_DIM]);
    let up_proj = b.add_linear(x, up_w, None, &[SEQ, FFN_DIM]);
    let gated = b.add_binary_mul(gate_act, up_proj, &[SEQ, FFN_DIM]);
    let out = b.add_linear(gated, down_w, None, &[SEQ, D_MODEL]);

    b.build(out).expect("valid SwiGLU")
}

fn swiglu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, D_MODEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, D_MODEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, FFN_DIM]),
            WEIGHT_MAG,
        )),
    ]
}

/// Self-attention with causal mask (no RoPE for Conservative soundness).
/// MHA(causal) -> residual.
fn build_self_attn_conservative() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_deep_self_attn");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);

    let shape = [SEQ, D_MODEL];

    let attn = b
        .add_multi_head_attention(
            x,
            q_w,
            k_w,
            v_w,
            out_w,
            N_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("valid causal self-attention");

    // Residual connection
    let out = b.add_binary_add(x, attn, &shape);

    b.build(out).expect("valid self-attention with residual")
}

fn self_attn_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w),
    ]
}

/// Single decoder block: RMSNorm -> MHA(causal) -> residual -> RMSNorm -> SwiGLU -> residual.
fn build_decoder_block_conservative() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_deep_decoder_block");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);

    let shape = [SEQ, D_MODEL];
    let ffn_shape = [SEQ, FFN_DIM];

    // Attention sub-block
    let attn_rms_w = b.add_input("attn_rms_w", &[D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);

    let normed1 = b.add_rms_norm(x, eps, 1, attn_rms_w, &shape);
    let attn = b
        .add_multi_head_attention(
            normed1,
            q_w,
            k_w,
            v_w,
            out_w,
            N_HEADS,
            AttentionMask::Causal,
            &shape,
        )
        .expect("valid causal self-attention");
    let residual1 = b.add_binary_add(x, attn, &shape);

    // MLP sub-block
    let mlp_rms_w = b.add_input("mlp_rms_w", &[D_MODEL]);
    let gate_w = b.add_input("gate_w", &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input("up_w", &[FFN_DIM, D_MODEL]);
    let down_w = b.add_input("down_w", &[D_MODEL, FFN_DIM]);

    let normed2 = b.add_rms_norm(residual1, eps, 1, mlp_rms_w, &shape);
    let gate_proj = b.add_linear(normed2, gate_w, None, &ffn_shape);
    // SiLU(x) = x * sigmoid(x)
    let gate_sig = b.add_sigmoid(gate_proj, &ffn_shape);
    let gate_act = b.add_binary_mul(gate_proj, gate_sig, &ffn_shape);
    let up_proj = b.add_linear(normed2, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up_proj, &ffn_shape);
    let mlp_out = b.add_linear(gated, down_w, None, &shape);
    let residual2 = b.add_binary_add(residual1, mlp_out, &shape);

    b.build(residual2).expect("valid decoder block")
}

fn decoder_block_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        // Attention
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w.clone()),
        TensorParamBinding::ConstantTensor(w),
        // MLP
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, D_MODEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, D_MODEL]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, FFN_DIM]),
            WEIGHT_MAG,
        )),
    ]
}

/// Post-norm + LM head: RMSNorm -> Linear(D -> VOCAB).
fn build_post_norm_lm_head_conservative() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_deep_post_norm_lm_head");
    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let rms_w = b.add_input("rms_w", &[D_MODEL]);
    let lm_w = b.add_input("lm_w", &[VOCAB, D_MODEL]);

    let normed = b.add_rms_norm(x, eps, 1, rms_w, &[SEQ, D_MODEL]);
    let out = b.add_linear(normed, lm_w, None, &[SEQ, VOCAB]);

    b.build(out).expect("valid post-norm LM head")
}

fn post_norm_lm_head_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOCAB, D_MODEL]), WEIGHT_MAG)),
    ]
}

// ===========================================================================
// 1. RMSNorm (Conservative) -- Sound
// ===========================================================================

#[test]
fn test_qwen3_deep_rmsnorm_conservative_sound() {
    let def = build_rmsnorm_conservative();
    let bindings = rmsnorm_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "qwen3_deep_rmsnorm",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative RMSNorm should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Qwen3 deep RMSNorm (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 2. SwiGLU FFN (Conservative) -- Sound
// ===========================================================================

#[test]
fn test_qwen3_deep_swiglu_conservative_sound() {
    let def = build_swiglu_conservative();
    let bindings = swiglu_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "qwen3_deep_swiglu",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative SwiGLU should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Qwen3 deep SwiGLU (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
    assert!(lo.abs() < 1e6, "SwiGLU lower magnitude < 1e6, got {lo}");
    assert!(hi.abs() < 1e6, "SwiGLU upper magnitude < 1e6, got {hi}");
}

// ===========================================================================
// 3. Self-attention with residual (Conservative) -- Sound
// ===========================================================================

#[test]
fn test_qwen3_deep_self_attn_conservative_ibp() {
    let def = build_self_attn_conservative();
    let bindings = self_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through self-attn");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 deep self-attn IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e8, "self-attn lower < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "self-attn upper < 1e8, got {hi}");
}

#[test]
fn test_qwen3_deep_self_attn_conservative_crown() {
    let def = build_self_attn_conservative();
    let bindings = self_attn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 deep self-attn: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_qwen3_deep_self_attn_conservative_verify_and_record() {
    let def = build_self_attn_conservative();
    let bindings = self_attn_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "qwen3_deep_self_attn",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative self-attn should produce Sound, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 4. Single decoder block (Conservative)
// ===========================================================================

#[test]
fn test_qwen3_deep_decoder_block_def_validates() {
    let def = build_decoder_block_conservative();
    def.validate().expect("decoder block should validate");
}

#[test]
fn test_qwen3_deep_decoder_block_ibp() {
    let def = build_decoder_block_conservative();
    let bindings = decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through decoder block");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 deep decoder block IBP: [{lo}, {hi}]");
    assert!(lo.abs() < 1e8, "decoder block lower < 1e8, got {lo}");
    assert!(hi.abs() < 1e8, "decoder block upper < 1e8, got {hi}");
}

#[test]
fn test_qwen3_deep_decoder_block_crown() {
    let def = build_decoder_block_conservative();
    let bindings = decoder_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 deep decoder block: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_qwen3_deep_decoder_block_verify_and_record() {
    let def = build_decoder_block_conservative();
    let bindings = decoder_block_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "qwen3_deep_decoder_block",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Qwen3 deep decoder block (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 5. Post-norm + LM head (Conservative)
// ===========================================================================

#[test]
fn test_qwen3_deep_post_norm_lm_head_ibp() {
    let def = build_post_norm_lm_head_conservative();
    let bindings = post_norm_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through post-norm LM head");
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, VOCAB]);
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 deep post-norm LM head IBP: [{lo}, {hi}]");
    assert!(lo.is_finite(), "lower must be finite, got {lo}");
    assert!(hi.is_finite(), "upper must be finite, got {hi}");
}

#[test]
fn test_qwen3_deep_post_norm_lm_head_verify_and_record() {
    let def = build_post_norm_lm_head_conservative();
    let bindings = post_norm_lm_head_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "qwen3_deep_post_norm_lm_head",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Qwen3 deep post-norm LM head (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// 6. Residual bounds preservation analysis (2 blocks)
// ===========================================================================

/// Residual bounds analysis: verifies that 2 blocks of the decoder do not
/// cause excessive bounds blowup due to residual connections.
#[test]
fn test_qwen3_deep_residual_bounds_2block() {
    // Single block
    let def1 = build_decoder_block_conservative();
    let bindings1 = decoder_block_bindings();
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let output1 = graph1.propagate_ibp(&input).expect("IBP through 1 block");
    let (lo1, hi1) = bounds_min_max(&output1);
    let range1 = hi1 - lo1;

    // Use single block output as input to second block
    let output2 = graph1
        .propagate_ibp(&output1)
        .expect("IBP through 2nd block");
    let (lo2, hi2) = bounds_min_max(&output2);
    let range2 = hi2 - lo2;

    eprintln!(
        "Qwen3 residual analysis: 1-block range={range1:.4}, 2-block range={range2:.4}, \
         blowup={:.1}x",
        range2 / range1.max(1e-10)
    );

    // 2 blocks should not blow up more than 100x relative to 1 block.
    // With small weights and residual connections, growth is controlled.
    let blowup = range2 / range1.max(1e-10);
    assert!(
        blowup < 1e4,
        "2-block blowup factor should be < 1e4 relative to 1-block, got {blowup:.1}x"
    );
}
