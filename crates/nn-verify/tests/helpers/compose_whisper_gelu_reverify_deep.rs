// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep Whisper GeLU re-verification: decoder FFN, stacked encoder layers, and
//! GeLU-erf vs GeLU-tanh CROWN comparison after NY CROWN relaxation fix.
//!
//! Extends `compose_whisper_gelu_reverify.rs` with patterns not covered there:
//!
//! 1. **Decoder FFN with GeLU-erf**: CROWN through the decoder-side FFN (Whisper
//!    uses `gelu_erf`, not the tanh approximation).
//!
//! 2. **GeLU-erf vs GeLU-tanh CROWN comparison**: Verifies both variants produce
//!    sound bounds and compares their CROWN tightness at multiple input ranges.
//!
//! 3. **Stacked encoder GeLU layers**: Two encoder blocks stacked, testing CROWN
//!    bound accumulation through multiple GeLU activations.
//!
//! Part of #4314: Re-verify GeLU models after NY CROWN relaxation fix.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    uniform_bounds, verify_and_assert, verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding, VerifyConfig};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Embedding / model dimension.
const D_MODEL: usize = 16;
/// FFN intermediate dimension: 4x the embedding dimension per Whisper spec.
const FFN_DIM: usize = 64;
/// Decoder sequence length (number of output tokens).
const DEC_SEQ: usize = 4;
/// Number of mel frequency bins (production: 128).
const N_MEL: usize = 4;
/// Encoder input sequence length of mel frames (production: 3000).
const MEL_SEQ_LEN: usize = 8;
/// Conv1d kernel size for encoder stems.
const CONV_KERNEL: usize = 3;
/// Conv1d padding for encoder stems.
const CONV_PADDING: usize = 1;
/// Number of attention heads (head_dim = D_MODEL / N_HEADS = 4).
const N_HEADS: usize = 4;
/// Small weight magnitude for bounded verification.
const W_MAG: f32 = 0.02;

fn after_conv1() -> usize {
    conv1d_out_len(MEL_SEQ_LEN, CONV_KERNEL, 1, CONV_PADDING)
}

fn after_conv2() -> usize {
    conv1d_out_len(after_conv1(), CONV_KERNEL, 2, CONV_PADDING)
}

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// ===========================================================================
// 1. Decoder FFN with GeLU-erf -- CROWN verification
// ===========================================================================

/// Build decoder FFN: LayerNorm -> Linear(D, 4D) -> GeLU-erf -> Linear(4D, D) -> residual.
///
/// Input: `[DEC_SEQ, D_MODEL]` (Variable).
/// Output: `[DEC_SEQ, D_MODEL]`.
///
/// The decoder FFN differs from encoder FFN in that it receives output from
/// cross-attention rather than self-attention, producing a different activation
/// distribution at the GeLU input.
fn build_decoder_ffn_gelu_erf() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("whisper_dec_ffn_gelu_erf_reverify");

    let x = b.add_input("x", &[DEC_SEQ, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    let shape = [DEC_SEQ, D_MODEL];
    let ffn_shape = [DEC_SEQ, FFN_DIM];

    // Pre-norm: LayerNorm
    let normed = b.add_layer_norm(x, eps, 1, ln_w, ln_b, &shape);
    // FFN: Linear(D, FFN_DIM) -> GeLU-erf -> Linear(FFN_DIM, D)
    let h = b.add_linear(normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu_erf(h, &ffn_shape);
    let proj = b.add_linear(act, ffn2_w, None, &shape);
    // Residual connection
    let out = b.add_binary_add(x, proj, &shape);

    b.build(out).expect("valid decoder FFN+GeLU-erf kernel")
}

fn decoder_ffn_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, FFN_DIM]), W_MAG)),
    ]
}

#[test]
fn test_whisper_dec_ffn_gelu_erf_crown_after_fix() {
    let def = build_decoder_ffn_gelu_erf();
    let bindings = decoder_ffn_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[DEC_SEQ, D_MODEL], 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("Whisper decoder FFN+GeLU-erf IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    // CROWN with fixed relaxation
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "Whisper decoder FFN+GeLU-erf CROWN: method={method:?}, [{crown_lo}, {crown_hi}], \
         width={crown_width}"
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    // Output bounds should be finite with small weights
    assert!(ibp_lo.is_finite() && ibp_hi.is_finite());
    assert!(crown_lo.is_finite() && crown_hi.is_finite());
}

// ===========================================================================
// 2. GeLU-erf vs GeLU-tanh CROWN comparison
// ===========================================================================

/// Build isolated GeLU-erf on the same shape as GeLU-tanh for comparison.
fn build_gelu_erf_isolated() -> TensorKernelDef {
    let t = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_gelu_erf_isolated_reverify");
    let x = b.add_input("x", &[t, D_MODEL]);
    let out = b.add_gelu_erf(x, &[t, D_MODEL]);
    b.build(out).expect("valid GeLU-erf isolated kernel")
}

/// Build isolated GeLU-tanh on the same shape for comparison.
fn build_gelu_tanh_isolated() -> TensorKernelDef {
    let t = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_gelu_tanh_isolated_reverify");
    let x = b.add_input("x", &[t, D_MODEL]);
    let out = b.add_gelu(x, &[t, D_MODEL]);
    b.build(out).expect("valid GeLU-tanh isolated kernel")
}

/// Compare CROWN bounds for GeLU-erf vs GeLU-tanh at multiple input ranges.
///
/// Whisper uses GeLU-erf. The CROWN relaxation fix may produce different
/// tightness for the two GeLU variants. This test verifies both produce
/// sound CROWN bounds and reports the relative tightness.
#[test]
fn test_whisper_gelu_erf_vs_tanh_crown_tightness() {
    let t = after_conv2();
    let erf_def = build_gelu_erf_isolated();
    let tanh_def = build_gelu_tanh_isolated();
    let erf_bindings = vec![TensorParamBinding::Variable];
    let tanh_bindings = vec![TensorParamBinding::Variable];
    let erf_graph = tensor_kernel_to_graph(&erf_def, &erf_bindings).expect("erf graph");
    let tanh_graph = tensor_kernel_to_graph(&tanh_def, &tanh_bindings).expect("tanh graph");

    for range in [0.5_f32, 1.0, 2.0, 4.0] {
        let input = uniform_bounds(&[t, D_MODEL], range);

        // GeLU-erf IBP + CROWN
        let erf_ibp = erf_graph.propagate_ibp(&input).expect("erf IBP");
        let (erf_ibp_lo, erf_ibp_hi) = bounds_min_max(&erf_ibp);
        let erf_ibp_width = erf_ibp_hi - erf_ibp_lo;

        let (erf_method, erf_crown, _) = assert_crown_tighter_when_not_fallback(&erf_graph, &input);
        let (erf_crown_lo, erf_crown_hi) = bounds_min_max(&erf_crown);
        let erf_crown_width = erf_crown_hi - erf_crown_lo;

        // GeLU-tanh IBP + CROWN
        let tanh_ibp = tanh_graph.propagate_ibp(&input).expect("tanh IBP");
        let (tanh_ibp_lo, tanh_ibp_hi) = bounds_min_max(&tanh_ibp);
        let tanh_ibp_width = tanh_ibp_hi - tanh_ibp_lo;

        let (tanh_method, tanh_crown, _) =
            assert_crown_tighter_when_not_fallback(&tanh_graph, &input);
        let (tanh_crown_lo, tanh_crown_hi) = bounds_min_max(&tanh_crown);
        let tanh_crown_width = tanh_crown_hi - tanh_crown_lo;

        eprintln!(
            "GeLU range={range}: erf IBP w={erf_ibp_width:.4}, \
             erf CROWN w={erf_crown_width:.4} ({erf_method:?})"
        );
        eprintln!(
            "             tanh IBP w={tanh_ibp_width:.4}, \
             tanh CROWN w={tanh_crown_width:.4} ({tanh_method:?})"
        );

        // Both variants must produce finite, valid bounds
        assert!(erf_ibp_lo.is_finite() && erf_ibp_hi.is_finite());
        assert!(erf_crown_lo.is_finite() && erf_crown_hi.is_finite());
        assert!(tanh_ibp_lo.is_finite() && tanh_ibp_hi.is_finite());
        assert!(tanh_crown_lo.is_finite() && tanh_crown_hi.is_finite());
        assert!(erf_crown_width >= 0.0, "erf CROWN width non-negative");
        assert!(tanh_crown_width >= 0.0, "tanh CROWN width non-negative");
    }
}

// ===========================================================================
// 3. Stacked encoder with GeLU -- CROWN through multiple GeLU layers
// ===========================================================================

/// Build two stacked encoder FFN blocks, each: LayerNorm -> Linear -> GeLU -> Linear -> residual.
///
/// Tests CROWN bound accumulation through two sequential GeLU activations.
fn build_stacked_encoder_gelu() -> (TensorKernelDef, usize) {
    let t = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_stacked_enc_gelu_reverify");

    let x = b.add_input("x", &[t, D_MODEL]);
    let eps1 = b.add_input("eps1", &[1]);
    let ln1_w = b.add_input("ln1_w", &[D_MODEL]);
    let ln1_b = b.add_input("ln1_b", &[D_MODEL]);
    let ffn1a_w = b.add_input("ffn1a_w", &[FFN_DIM, D_MODEL]);
    let ffn1b_w = b.add_input("ffn1b_w", &[D_MODEL, FFN_DIM]);
    let eps2 = b.add_input("eps2", &[1]);
    let ln2_w = b.add_input("ln2_w", &[D_MODEL]);
    let ln2_b = b.add_input("ln2_b", &[D_MODEL]);
    let ffn2a_w = b.add_input("ffn2a_w", &[FFN_DIM, D_MODEL]);
    let ffn2b_w = b.add_input("ffn2b_w", &[D_MODEL, FFN_DIM]);

    let shape = [t, D_MODEL];
    let ffn_shape = [t, FFN_DIM];

    // Block 1
    let n1 = b.add_layer_norm(x, eps1, 1, ln1_w, ln1_b, &shape);
    let h1 = b.add_linear(n1, ffn1a_w, None, &ffn_shape);
    let a1 = b.add_gelu(h1, &ffn_shape);
    let p1 = b.add_linear(a1, ffn1b_w, None, &shape);
    let r1 = b.add_binary_add(x, p1, &shape);

    // Block 2
    let n2 = b.add_layer_norm(r1, eps2, 1, ln2_w, ln2_b, &shape);
    let h2 = b.add_linear(n2, ffn2a_w, None, &ffn_shape);
    let a2 = b.add_gelu(h2, &ffn_shape);
    let p2 = b.add_linear(a2, ffn2b_w, None, &shape);
    let out = b.add_binary_add(r1, p2, &shape);

    (b.build(out).expect("valid stacked encoder GeLU kernel"), t)
}

fn stacked_encoder_bindings() -> Vec<TensorParamBinding> {
    let block = || -> Vec<TensorParamBinding> {
        vec![
            TensorParamBinding::ConstantScalar(1e-5),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[FFN_DIM, D_MODEL]),
                W_MAG,
            )),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[D_MODEL, FFN_DIM]),
                W_MAG,
            )),
        ]
    };
    let mut bindings = vec![TensorParamBinding::Variable];
    bindings.extend(block());
    bindings.extend(block());
    bindings
}

#[test]
fn test_whisper_stacked_enc_gelu_crown_after_fix() {
    let (def, t) = build_stacked_encoder_gelu();
    let bindings = stacked_encoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    eprintln!(
        "Whisper stacked encoder (2 GeLU blocks) IBP: [{ibp_lo}, {ibp_hi}], \
         width={:.4}",
        ibp_hi - ibp_lo
    );

    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!(
        "Whisper stacked encoder (2 GeLU blocks) CROWN: method={method:?}, \
         [{crown_lo}, {crown_hi}], width={:.4}",
        crown_hi - crown_lo
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    assert!(ibp_lo.is_finite() && ibp_hi.is_finite());
    assert!(crown_lo.is_finite() && crown_hi.is_finite());
}

// ===========================================================================
// 4. Recording tests
// ===========================================================================

/// Record decoder FFN+GeLU-erf CROWN verification to status file.
#[test]
fn test_whisper_dec_ffn_gelu_erf_record_crown_reverify() {
    let def = build_decoder_ffn_gelu_erf();
    let bindings = decoder_ffn_bindings();
    let input = uniform_bounds(&[DEC_SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "whisper_dec_ffn_gelu_erf_crown_reverify",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "RECORD whisper_dec_ffn_gelu_erf_crown_reverify: [{lo}, {hi}], width={:.4}, \
         method={:?}, soundness={:?}",
        hi - lo,
        result.verification.method,
        result.verification.soundness_mode
    );
}

/// Record GeLU-erf isolated CROWN verification to status file.
#[test]
fn test_whisper_gelu_erf_isolated_record_crown_reverify() {
    let def = build_gelu_erf_isolated();
    let t = after_conv2();
    let bindings = vec![TensorParamBinding::Variable];
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "whisper_gelu_erf_isolated_crown_reverify",
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "RECORD whisper_gelu_erf_isolated_crown_reverify: [{lo}, {hi}], width={:.4}, \
         method={:?}, soundness={:?}",
        hi - lo,
        result.verification.method,
        result.verification.soundness_mode
    );
}

/// Record stacked encoder (2 GeLU blocks) CROWN verification.
#[test]
fn test_whisper_stacked_enc_gelu_record_crown_reverify() {
    let (def, t) = build_stacked_encoder_gelu();
    let bindings = stacked_encoder_bindings();
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "whisper_stacked_enc_gelu_crown_reverify",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "RECORD whisper_stacked_enc_gelu_crown_reverify: [{lo}, {hi}], width={:.4}, \
         method={:?}, soundness={:?}",
        hi - lo,
        result.verification.method,
        result.verification.soundness_mode
    );
}
