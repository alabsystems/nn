// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper mel spectrogram + encoder-decoder bridge compose verification.
//!
//! Tests bounds propagation through Whisper sub-components not yet covered:
//!
//! 1. **Mel spectrogram linear projection**: Projection from mel bins to model dim
//!    without GELU, testing pure linear operator bounds.
//! 2. **Multi-layer encoder stack (3 layers)**: Tests bounds stability through
//!    deeper encoder stacking with chained LayerNorms.
//! 3. **Encoder-decoder bridge**: Full encoder output (constant) feeding into
//!    2-block decoder with final LayerNorm + output projection (LM head).
//! 4. **Narrow input monotonicity on encoder stack**: Verifies narrower input
//!    produces tighter output bounds through the deeper encoder.
//!
//! Part of compose verification deepening for Whisper model.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    uniform_bounds, verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding, VerificationSoundnessMode};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const N_MEL: usize = 4;
const SEQ_LEN: usize = 8;
const EMBED_DIM: usize = 32;
const NUM_HEADS: usize = 4;
const FFN_DIM: usize = 128;
const CONV_KERNEL: usize = 3;
const CONV_PADDING: usize = 1;
const DEC_SEQ_LEN: usize = 4;
const ENC_SEQ_LEN: usize = 4; // after conv2 output
const VOCAB_SIZE: usize = 16;
const WEIGHT_MAG: f32 = 0.02;

fn after_conv1_len() -> usize {
    conv1d_out_len(SEQ_LEN, CONV_KERNEL, 1, CONV_PADDING)
}

fn after_conv2_len() -> usize {
    conv1d_out_len(after_conv1_len(), CONV_KERNEL, 2, CONV_PADDING)
}

// ---------------------------------------------------------------------------
// 1. Mel spectrogram linear projection
// ---------------------------------------------------------------------------

/// Build a mel spectrogram linear projection: Linear(N_MEL -> EMBED_DIM).
///
/// Input: `[N_MEL, SEQ_LEN]` (Variable).
/// Output: `[EMBED_DIM, SEQ_LEN]` via Conv1d(k=1, s=1, p=0) which is
/// equivalent to a per-frame linear projection.
fn build_mel_projection_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_mel_projection");

    let mel = b.add_input("mel", &[N_MEL, SEQ_LEN]);
    let proj_w = b.add_input("proj_weight", &[EMBED_DIM, N_MEL, 1]);
    let proj_b = b.add_input("proj_bias", &[EMBED_DIM]);

    let out = b.add_conv1d(mel, proj_w, Some(proj_b), 1, 0, &[EMBED_DIM, SEQ_LEN]);

    b.build(out).expect("valid mel projection kernel")
}

fn mel_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[EMBED_DIM, N_MEL, 1]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[EMBED_DIM]), 0.0f32)),
    ]
}

// ---------------------------------------------------------------------------
// 2. 3-layer encoder stack
// ---------------------------------------------------------------------------

/// Build a 3-layer stacked encoder (deeper than existing 1-layer and 2-layer tests).
///
/// Input: `[T, EMBED_DIM]` (Variable, after conv + transpose + pos_emb).
/// Output: `[T, EMBED_DIM]` after 3 transformer blocks + final LayerNorm.
fn build_3_layer_encoder_stack() -> TensorKernelDef {
    let t_out = after_conv2_len();
    let mut b = TensorBlockBuilder::new("whisper_enc_3layer_stack");

    let input = b.add_input("x", &[t_out, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    let shape = [t_out, EMBED_DIM];
    let ffn_shape = [t_out, FFN_DIM];

    let mut current = input;

    for block_idx in 0..3 {
        let pfx = format!("b{block_idx}");

        let sa_ln_w = b.add_input(&format!("{pfx}_sa_ln_w"), &[EMBED_DIM]);
        let sa_ln_b = b.add_input(&format!("{pfx}_sa_ln_b"), &[EMBED_DIM]);
        let q_w = b.add_input(&format!("{pfx}_qw"), &[EMBED_DIM, EMBED_DIM]);
        let k_w = b.add_input(&format!("{pfx}_kw"), &[EMBED_DIM, EMBED_DIM]);
        let v_w = b.add_input(&format!("{pfx}_vw"), &[EMBED_DIM, EMBED_DIM]);
        let out_w = b.add_input(&format!("{pfx}_ow"), &[EMBED_DIM, EMBED_DIM]);
        let ffn_ln_w = b.add_input(&format!("{pfx}_ffn_ln_w"), &[EMBED_DIM]);
        let ffn_ln_b = b.add_input(&format!("{pfx}_ffn_ln_b"), &[EMBED_DIM]);
        let ffn1_w = b.add_input(&format!("{pfx}_ffn1w"), &[FFN_DIM, EMBED_DIM]);
        let ffn2_w = b.add_input(&format!("{pfx}_ffn2w"), &[EMBED_DIM, FFN_DIM]);

        // Pre-norm self-attention
        let sa_normed = b.add_layer_norm(current, eps, 1, sa_ln_w, sa_ln_b, &shape);
        let sa_out = b
            .add_multi_head_attention(
                sa_normed,
                q_w,
                k_w,
                v_w,
                out_w,
                NUM_HEADS,
                AttentionMask::Standard,
                &shape,
            )
            .expect("valid encoder self-attention");
        let residual1 = b.add_binary_add(current, sa_out, &shape);

        // Pre-norm FFN
        let ffn_normed = b.add_layer_norm(residual1, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
        let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
        let act = b.add_gelu(ffn1, &ffn_shape);
        let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
        current = b.add_binary_add(residual1, ffn2, &shape);
    }

    // Final LayerNorm
    let final_ln_w = b.add_input("final_ln_w", &[EMBED_DIM]);
    let final_ln_b = b.add_input("final_ln_b", &[EMBED_DIM]);
    let final_eps = b.add_input("final_eps", &[1]);
    let output = b.add_layer_norm(current, final_eps, 1, final_ln_w, final_ln_b, &shape);

    b.build(output).expect("valid 3-layer encoder stack")
}

fn three_layer_encoder_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
    ];

    for _ in 0..3 {
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // sa_ln_w
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // sa_ln_b
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone())); // q_w
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone())); // k_w
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone())); // v_w
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone())); // out_w
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone())); // ffn_ln_w
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone())); // ffn_ln_b
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn1.clone())); // ffn1_w
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn2.clone())); // ffn2_w
    }

    // Final LayerNorm
    bindings.push(TensorParamBinding::ConstantTensor(ln_w));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    bindings
}

// ---------------------------------------------------------------------------
// 3. Encoder-decoder bridge (encoder output -> 2-block decoder -> LM head)
// ---------------------------------------------------------------------------

/// Build an encoder-decoder bridge: encoder output (constant) feeds into
/// 2-block decoder with final LayerNorm + Linear (LM head).
///
/// Input: `[DEC_SEQ_LEN, EMBED_DIM]` (Variable -- token embeddings).
/// Encoder output: `[ENC_SEQ_LEN, EMBED_DIM]` (Constant).
/// Output: `[DEC_SEQ_LEN, VOCAB_SIZE]` (logits).
fn build_encoder_decoder_bridge() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_enc_dec_bridge");

    let dec_input = b.add_input("dec_input", &[DEC_SEQ_LEN, EMBED_DIM]);
    let enc_output = b.add_input("enc_output", &[ENC_SEQ_LEN, EMBED_DIM]);
    let eps = b.add_input("eps", &[1]);

    let shape = [DEC_SEQ_LEN, EMBED_DIM];
    let ffn_shape = [DEC_SEQ_LEN, FFN_DIM];

    let mut current = dec_input;

    for block_idx in 0..2 {
        let pfx = format!("d{block_idx}");

        // Self-attention weights
        let sa_ln_w = b.add_input(&format!("{pfx}_sa_ln_w"), &[EMBED_DIM]);
        let sa_ln_b = b.add_input(&format!("{pfx}_sa_ln_b"), &[EMBED_DIM]);
        let sa_q_w = b.add_input(&format!("{pfx}_sa_qw"), &[EMBED_DIM, EMBED_DIM]);
        let sa_k_w = b.add_input(&format!("{pfx}_sa_kw"), &[EMBED_DIM, EMBED_DIM]);
        let sa_v_w = b.add_input(&format!("{pfx}_sa_vw"), &[EMBED_DIM, EMBED_DIM]);
        let sa_out_w = b.add_input(&format!("{pfx}_sa_ow"), &[EMBED_DIM, EMBED_DIM]);

        // Cross-attention weights
        let ca_ln_w = b.add_input(&format!("{pfx}_ca_ln_w"), &[EMBED_DIM]);
        let ca_ln_b = b.add_input(&format!("{pfx}_ca_ln_b"), &[EMBED_DIM]);
        let ca_q_w = b.add_input(&format!("{pfx}_ca_qw"), &[EMBED_DIM, EMBED_DIM]);
        let ca_k_w = b.add_input(&format!("{pfx}_ca_kw"), &[EMBED_DIM, EMBED_DIM]);
        let ca_v_w = b.add_input(&format!("{pfx}_ca_vw"), &[EMBED_DIM, EMBED_DIM]);
        let ca_out_w = b.add_input(&format!("{pfx}_ca_ow"), &[EMBED_DIM, EMBED_DIM]);

        // FFN weights
        let ffn_ln_w = b.add_input(&format!("{pfx}_ffn_ln_w"), &[EMBED_DIM]);
        let ffn_ln_b = b.add_input(&format!("{pfx}_ffn_ln_b"), &[EMBED_DIM]);
        let ffn1_w = b.add_input(&format!("{pfx}_ffn1w"), &[FFN_DIM, EMBED_DIM]);
        let ffn2_w = b.add_input(&format!("{pfx}_ffn2w"), &[EMBED_DIM, FFN_DIM]);

        // Sub-block 1: Causal self-attention
        let sa_normed = b.add_layer_norm(current, eps, 1, sa_ln_w, sa_ln_b, &shape);
        let sa_out = b
            .add_multi_head_attention(
                sa_normed,
                sa_q_w,
                sa_k_w,
                sa_v_w,
                sa_out_w,
                NUM_HEADS,
                AttentionMask::Causal,
                &shape,
            )
            .expect("valid causal self-attention");
        let residual1 = b.add_binary_add(current, sa_out, &shape);

        // Sub-block 2: Cross-attention with encoder output
        let ca_normed = b.add_layer_norm(residual1, eps, 1, ca_ln_w, ca_ln_b, &shape);
        let ca_out = b
            .add_multi_head_cross_attention(
                ca_normed,
                enc_output,
                ca_q_w,
                ca_k_w,
                ca_v_w,
                ca_out_w,
                NUM_HEADS,
                AttentionMask::Standard,
                &shape,
            )
            .expect("valid cross-attention");
        let residual2 = b.add_binary_add(residual1, ca_out, &shape);

        // Sub-block 3: FFN
        let ffn_normed = b.add_layer_norm(residual2, eps, 1, ffn_ln_w, ffn_ln_b, &shape);
        let ffn1 = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
        let act = b.add_gelu(ffn1, &ffn_shape);
        let ffn2 = b.add_linear(act, ffn2_w, None, &shape);
        current = b.add_binary_add(residual2, ffn2, &shape);
    }

    // Final LayerNorm
    let final_ln_w = b.add_input("final_ln_w", &[EMBED_DIM]);
    let final_ln_b = b.add_input("final_ln_b", &[EMBED_DIM]);
    let final_eps = b.add_input("final_eps", &[1]);
    let normed = b.add_layer_norm(current, final_eps, 1, final_ln_w, final_ln_b, &shape);

    // LM head: Linear(EMBED_DIM -> VOCAB_SIZE)
    let lm_head_w = b.add_input("lm_head_w", &[VOCAB_SIZE, EMBED_DIM]);
    let output = b.add_linear(normed, lm_head_w, None, &[DEC_SEQ_LEN, VOCAB_SIZE]);

    b.build(output).expect("valid encoder-decoder bridge")
}

fn encoder_decoder_bridge_bindings() -> Vec<TensorParamBinding> {
    let d = EMBED_DIM;
    let enc_const = ArrayD::from_elem(IxDyn(&[ENC_SEQ_LEN, d]), 0.1f32);
    let w_proj = ArrayD::from_elem(IxDyn(&[d, d]), WEIGHT_MAG);
    let w_ffn1 = ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), WEIGHT_MAG);
    let w_ffn2 = ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), WEIGHT_MAG);
    let ln_w = ArrayD::from_elem(IxDyn(&[d]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[d]), 0.0f32);

    let mut bindings = vec![
        TensorParamBinding::Variable,                  // dec_input
        TensorParamBinding::ConstantTensor(enc_const), // enc_output
        TensorParamBinding::ConstantScalar(1e-5),      // eps
    ];

    for _ in 0..2 {
        // Self-attention
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));

        // Cross-attention
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_proj.clone()));

        // FFN
        bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn1.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ffn2.clone()));
    }

    // Final LayerNorm
    bindings.push(TensorParamBinding::ConstantTensor(ln_w));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // LM head
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[VOCAB_SIZE, d]),
        WEIGHT_MAG,
    )));

    bindings
}

// ===========================================================================
// Tests: Mel spectrogram linear projection
// ===========================================================================

#[test]
fn test_whisper_mel_projection_def_validates() {
    let def = build_mel_projection_kernel();
    def.validate().expect("mel projection should validate");
}

#[test]
fn test_whisper_mel_projection_ibp_propagates() {
    let def = build_mel_projection_kernel();
    let bindings = mel_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through mel projection");
    assert_eq!(output.lower_upper().0.shape(), &[EMBED_DIM, SEQ_LEN]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper mel projection IBP: bounds=[{lo_min}, {hi_max}]");

    // Pure linear: Conv1d(k=1) with small weights. Tight bounds expected.
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(lo_min < hi_max, "bounds must be non-degenerate");
}

#[test]
fn test_whisper_mel_projection_crown_propagation() {
    let def = build_mel_projection_kernel();
    let bindings = mel_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[EMBED_DIM, SEQ_LEN]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper mel projection: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

#[test]
fn test_whisper_mel_projection_verify_and_record() {
    let def = build_mel_projection_kernel();
    let bindings = mel_projection_bindings();
    let input = uniform_bounds(&[N_MEL, SEQ_LEN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_mel_projection");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[EMBED_DIM, SEQ_LEN]
    );
}

// ===========================================================================
// Tests: 3-layer encoder stack
// ===========================================================================

#[test]
fn test_whisper_3layer_enc_def_validates() {
    let def = build_3_layer_encoder_stack();
    def.validate().expect("3-layer encoder should validate");
}

#[test]
fn test_whisper_3layer_enc_graph_builds() {
    let def = build_3_layer_encoder_stack();
    let bindings = three_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // 3 blocks x (LN + MHA + residual + LN + FFN + residual) + final LN >= 60 nodes
    assert!(
        graph.num_nodes() >= 50,
        "3-layer encoder graph should have >= 50 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_whisper_3layer_enc_ibp_propagates() {
    let t_out = after_conv2_len();
    let def = build_3_layer_encoder_stack();
    let bindings = three_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[t_out, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 3-layer encoder");
    assert_eq!(output.lower_upper().0.shape(), &[t_out, EMBED_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper 3-layer encoder IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_whisper_3layer_enc_crown_propagation() {
    let t_out = after_conv2_len();
    let def = build_3_layer_encoder_stack();
    let bindings = three_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[t_out, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[t_out, EMBED_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper 3-layer encoder: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

#[test]
fn test_whisper_3layer_enc_verify_and_record() {
    let t_out = after_conv2_len();
    let def = build_3_layer_encoder_stack();
    let bindings = three_layer_encoder_bindings();
    let input = uniform_bounds(&[t_out, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_encoder_3layer_stack");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[t_out, EMBED_DIM]
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "3-layer encoder with LayerNorms should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

/// Narrower input produces tighter output bounds through 3-layer encoder.
#[test]
fn test_whisper_3layer_enc_narrow_inputs_tighter() {
    let t_out = after_conv2_len();
    let def = build_3_layer_encoder_stack();
    let bindings = three_layer_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[t_out, EMBED_DIM], 10.0);
    let narrow_input = uniform_bounds(&[t_out, EMBED_DIM], 1.0);

    let wide_output = graph.propagate_ibp(&wide_input).expect("wide IBP");
    let narrow_output = graph.propagate_ibp(&narrow_input).expect("narrow IBP");

    let (wide_lo, wide_hi) = wide_output.lower_upper();
    let (narrow_lo, narrow_hi) = narrow_output.lower_upper();

    let wide_range = wide_hi.iter().zip(wide_lo.iter()).map(|(h, l)| h - l);
    let narrow_range = narrow_hi.iter().zip(narrow_lo.iter()).map(|(h, l)| h - l);

    let tighter_count = wide_range.zip(narrow_range).filter(|(w, n)| n <= w).count();
    let total = wide_lo.len();
    assert!(
        tighter_count > total / 2,
        "narrow input should produce tighter bounds for > 50% of elements, got {tighter_count}/{total}"
    );
}

// ===========================================================================
// Tests: Encoder-decoder bridge
// ===========================================================================

#[test]
fn test_whisper_enc_dec_bridge_def_validates() {
    let def = build_encoder_decoder_bridge();
    def.validate()
        .expect("encoder-decoder bridge should validate");
}

#[test]
fn test_whisper_enc_dec_bridge_graph_builds() {
    let def = build_encoder_decoder_bridge();
    let bindings = encoder_decoder_bridge_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // 2 decoder blocks + final LN + LM head >= 50 nodes
    assert!(
        graph.num_nodes() >= 50,
        "encoder-decoder bridge graph should have >= 50 nodes, got {}",
        graph.num_nodes()
    );
}

#[test]
fn test_whisper_enc_dec_bridge_ibp_propagates() {
    let def = build_encoder_decoder_bridge();
    let bindings = encoder_decoder_bridge_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through encoder-decoder bridge");
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ_LEN, VOCAB_SIZE]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper encoder-decoder bridge IBP: bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_whisper_enc_dec_bridge_crown_propagation() {
    let def = build_encoder_decoder_bridge();
    let bindings = encoder_decoder_bridge_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[DEC_SEQ_LEN, VOCAB_SIZE]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Whisper encoder-decoder bridge: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

#[test]
fn test_whisper_enc_dec_bridge_verify_and_record() {
    let def = build_encoder_decoder_bridge();
    let bindings = encoder_decoder_bridge_bindings();
    let input = uniform_bounds(&[DEC_SEQ_LEN, EMBED_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_encoder_decoder_bridge");
    assert_eq!(result.num_variables, 1, "single Variable input (dec_input)");
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[DEC_SEQ_LEN, VOCAB_SIZE]
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "encoder-decoder bridge with LayerNorms should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
