// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper GeLU re-verification compose tests after NY CROWN relaxation
//! soundness fix (rev e810fb2b).
//!
//! Prior to the fix, all Whisper verification entries used IBP-only (IbpValidated).
//! IBP is unaffected by the CROWN relaxation bug, so existing IBP results remain
//! sound. However, CROWN linearization through GeLU was potentially unsound before
//! the fix. These tests explicitly exercise CROWN propagation through GeLU-using
//! Whisper sub-blocks to confirm:
//!
//! 1. **Isolated GeLU CROWN**: CROWN bounds through standalone GeLU are sound
//!    and tighter than IBP.
//!
//! 2. **Encoder FFN CROWN**: Linear -> GeLU -> Linear with residual. CROWN
//!    should produce tighter bounds than IBP through the GeLU activation.
//!
//! 3. **Encoder block with GeLU**: Full encoder block (self-attention + FFN)
//!    verified with CROWN. Tests composed CROWN through LayerNorm + GeLU.
//!
//! 4. **IBP vs CROWN tightness comparison**: Quantifies the improvement from
//!    using CROWN over IBP for GeLU-containing blocks.
//!
//! 5. **Mel feature extraction CROWN**: Conv1d -> GeLU -> Conv1d -> GeLU stem
//!    verified with CROWN to confirm CROWN soundness through conv+GeLU chains.
//!
//! Part of #4314: Re-verify GeLU models after NY CROWN relaxation fix.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, conv1d_out_len,
    uniform_bounds, verify_and_assert, verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::AttentionMask;
use nn_verify::{tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding, VerifyConfig};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Embedding / model dimension.
const D_MODEL: usize = 16;
/// Number of attention heads (head_dim = D_MODEL / N_HEADS = 4).
const N_HEADS: usize = 4;
/// FFN intermediate dimension: 4x the embedding dimension per Whisper spec.
const FFN_DIM: usize = 64;
/// Number of mel frequency bins (production: 128).
const N_MEL: usize = 4;
/// Encoder input sequence length of mel frames (production: 3000).
const MEL_SEQ_LEN: usize = 8;
/// Conv1d kernel size for encoder stems.
const CONV_KERNEL: usize = 3;
/// Conv1d padding for encoder stems.
const CONV_PADDING: usize = 1;
/// Small weight magnitude for bounded verification.
const W_MAG: f32 = 0.02;

/// Output sequence length after the first conv (stride=1, same padding).
fn after_conv1() -> usize {
    conv1d_out_len(MEL_SEQ_LEN, CONV_KERNEL, 1, CONV_PADDING)
}

/// Output sequence length after the second conv (stride=2, same padding).
fn after_conv2() -> usize {
    conv1d_out_len(after_conv1(), CONV_KERNEL, 2, CONV_PADDING)
}

// ===========================================================================
// 1. Isolated GeLU CROWN verification
// ===========================================================================

/// Build standalone GeLU activation on a 2D tensor.
///
/// Input: `[T, D_MODEL]` (Variable).
/// Output: `[T, D_MODEL]`.
fn build_gelu_isolated() -> (TensorKernelDef, usize) {
    let t = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_gelu_isolated_reverify");
    let x = b.add_input("x", &[t, D_MODEL]);
    let out = b.add_gelu(x, &[t, D_MODEL]);
    (b.build(out).expect("valid GeLU isolated kernel"), t)
}

#[test]
fn test_whisper_gelu_isolated_crown_after_fix() {
    let (def, t) = build_gelu_isolated();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    // Run IBP as baseline
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    eprintln!("Whisper GeLU isolated IBP: [{ibp_lo}, {ibp_hi}]");

    // Run CROWN -- should succeed and be tighter after the fix
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!("Whisper GeLU isolated CROWN: method={method:?}, [{crown_lo}, {crown_hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    // GeLU(-1) ~ -0.159, GeLU(1) ~ 0.841: bounds should be reasonable
    assert!(
        ibp_lo >= -0.5,
        "GeLU IBP lower should be >= -0.5, got {ibp_lo}"
    );
    assert!(
        ibp_hi <= 1.5,
        "GeLU IBP upper should be <= 1.5, got {ibp_hi}"
    );
}

// ===========================================================================
// 2. Encoder FFN with GeLU -- CROWN verification
// ===========================================================================

/// Build encoder FFN: LayerNorm -> Linear(D, 4D) -> GeLU -> Linear(4D, D) -> residual.
///
/// Input: `[T, D_MODEL]` (Variable).
/// Output: `[T, D_MODEL]`.
fn build_encoder_ffn_gelu() -> (TensorKernelDef, usize) {
    let t = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_enc_ffn_gelu_reverify");

    let x = b.add_input("x", &[t, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    let shape = [t, D_MODEL];
    let ffn_shape = [t, FFN_DIM];

    // Pre-norm: LayerNorm
    let normed = b.add_layer_norm(x, eps, 1, ln_w, ln_b, &shape);
    // FFN: Linear(D, FFN_DIM) -> GeLU -> Linear(FFN_DIM, D)
    let h = b.add_linear(normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(h, &ffn_shape);
    let proj = b.add_linear(act, ffn2_w, None, &shape);
    // Residual connection
    let out = b.add_binary_add(x, proj, &shape);

    (b.build(out).expect("valid encoder FFN+GeLU kernel"), t)
}

fn encoder_ffn_gelu_bindings() -> Vec<TensorParamBinding> {
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
fn test_whisper_enc_ffn_gelu_crown_after_fix() {
    let (def, t) = build_encoder_ffn_gelu();
    let bindings = encoder_ffn_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("Whisper encoder FFN+GeLU IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    // CROWN with fixed relaxation
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "Whisper encoder FFN+GeLU CROWN: method={method:?}, [{crown_lo}, {crown_hi}], width={crown_width}"
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    // Output bounds should be finite with small weights
    assert!(ibp_lo.is_finite() && ibp_hi.is_finite());
    assert!(crown_lo.is_finite() && crown_hi.is_finite());
}

// ===========================================================================
// 3. Encoder block (self-attention + FFN with GeLU) -- CROWN
// ===========================================================================

/// Build a full encoder block: self-attention + FFN with GeLU.
///
/// Input: `[T, D_MODEL]` (Variable).
/// Output: `[T, D_MODEL]`.
///
/// Architecture:
///   LayerNorm -> MHA(standard) -> + residual
///   LayerNorm -> Linear(D, 4D) -> GeLU -> Linear(4D, D) -> + residual
fn build_encoder_block_with_gelu() -> (TensorKernelDef, usize) {
    let t = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_enc_block_gelu_reverify");

    // Inputs
    let x = b.add_input("x", &[t, D_MODEL]);
    // Self-attention params
    let sa_eps = b.add_input("sa_eps", &[1]);
    let sa_ln_w = b.add_input("sa_ln_w", &[D_MODEL]);
    let sa_ln_b = b.add_input("sa_ln_b", &[D_MODEL]);
    let q_w = b.add_input("q_w", &[D_MODEL, D_MODEL]);
    let k_w = b.add_input("k_w", &[D_MODEL, D_MODEL]);
    let v_w = b.add_input("v_w", &[D_MODEL, D_MODEL]);
    let out_w = b.add_input("out_w", &[D_MODEL, D_MODEL]);
    // FFN params
    let ffn_eps = b.add_input("ffn_eps", &[1]);
    let ffn_ln_w = b.add_input("ffn_ln_w", &[D_MODEL]);
    let ffn_ln_b = b.add_input("ffn_ln_b", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    let shape = [t, D_MODEL];
    let ffn_shape = [t, FFN_DIM];

    // Sub-block 1: Self-attention with residual
    let sa_normed = b.add_layer_norm(x, sa_eps, 1, sa_ln_w, sa_ln_b, &shape);
    let attn = b
        .add_multi_head_attention(
            sa_normed,
            q_w,
            k_w,
            v_w,
            out_w,
            N_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid self-attention");
    let sa_out = b.add_binary_add(x, attn, &shape);

    // Sub-block 2: FFN with GeLU and residual
    let ffn_normed = b.add_layer_norm(sa_out, ffn_eps, 1, ffn_ln_w, ffn_ln_b, &shape);
    let h = b.add_linear(ffn_normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(h, &ffn_shape);
    let proj = b.add_linear(act, ffn2_w, None, &shape);
    let out = b.add_binary_add(sa_out, proj, &shape);

    (b.build(out).expect("valid encoder block+GeLU kernel"), t)
}

fn encoder_block_gelu_bindings() -> Vec<TensorParamBinding> {
    let d = D_MODEL;
    let w = ArrayD::from_elem(IxDyn(&[d, d]), W_MAG);
    vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // sa_eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)), // sa_ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)), // sa_ln_b
        TensorParamBinding::ConstantTensor(w.clone()), // q_w
        TensorParamBinding::ConstantTensor(w.clone()), // k_w
        TensorParamBinding::ConstantTensor(w.clone()), // v_w
        TensorParamBinding::ConstantTensor(w),    // out_w
        TensorParamBinding::ConstantScalar(1e-5), // ffn_eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 1.0f32)), // ffn_ln_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d]), 0.0f32)), // ffn_ln_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, d]), W_MAG)), // ffn1_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[d, FFN_DIM]), W_MAG)), // ffn2_w
    ]
}

#[test]
fn test_whisper_enc_block_gelu_crown_after_fix() {
    let (def, t) = build_encoder_block_with_gelu();
    let bindings = encoder_block_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("Whisper encoder block+GeLU IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    // CROWN with fixed relaxation
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "Whisper encoder block+GeLU CROWN: method={method:?}, [{crown_lo}, {crown_hi}], width={crown_width}"
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }
}

// ===========================================================================
// 4. IBP vs CROWN tightness comparison for GeLU paths
// ===========================================================================

/// Compare IBP and CROWN bounds width for the encoder FFN sub-block at
/// different input ranges. After the CROWN relaxation fix, CROWN should
/// produce tighter bounds than IBP for GeLU-containing blocks.
#[test]
fn test_whisper_gelu_ibp_vs_crown_tightness() {
    let (def, t) = build_encoder_ffn_gelu();
    let bindings = encoder_ffn_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    for range in [0.5_f32, 1.0, 2.0] {
        let input = uniform_bounds(&[t, D_MODEL], range);

        // IBP
        let ibp_output = graph.propagate_ibp(&input).expect("IBP");
        let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
        let ibp_width = ibp_hi - ibp_lo;

        // CROWN
        let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
        let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
        let crown_width = crown_hi - crown_lo;

        eprintln!(
            "GeLU FFN range={range}: IBP width={ibp_width:.4}, CROWN width={crown_width:.4} \
             (method={method:?}, ratio={:.2}x)",
            if crown_width > 0.0 {
                ibp_width / crown_width
            } else {
                f32::INFINITY
            }
        );

        // Both must produce finite, valid bounds
        assert!(ibp_lo.is_finite() && ibp_hi.is_finite());
        assert!(crown_lo.is_finite() && crown_hi.is_finite());
        assert!(ibp_width >= 0.0, "IBP width must be non-negative");
        assert!(crown_width >= 0.0, "CROWN width must be non-negative");
    }
}

// ===========================================================================
// 5. Mel feature extraction CROWN (Conv1d -> GeLU -> Conv1d -> GeLU)
// ===========================================================================

/// Build mel spectrogram feature extraction with GeLU activations.
///
/// Input: `[N_MEL, MEL_SEQ_LEN]` (Variable).
/// Output: `[D_MODEL, T_OUT]`.
fn build_mel_features_gelu() -> (TensorKernelDef, usize) {
    let t_mid = after_conv1();
    let t_out = after_conv2();
    let mut b = TensorBlockBuilder::new("whisper_mel_gelu_reverify");

    let mel = b.add_input("mel", &[N_MEL, MEL_SEQ_LEN]);

    // Conv stem #1: Conv1d(N_MEL -> D_MODEL, k=3, s=1, p=1) -> GeLU
    let c1_w = b.add_input("conv1_w", &[D_MODEL, N_MEL, CONV_KERNEL]);
    let c1_b = b.add_input("conv1_b", &[D_MODEL]);
    let c1 = b.add_conv1d(mel, c1_w, Some(c1_b), 1, CONV_PADDING, &[D_MODEL, t_mid]);
    let g1 = b.add_gelu(c1, &[D_MODEL, t_mid]);

    // Conv stem #2: Conv1d(D_MODEL -> D_MODEL, k=3, s=2, p=1) -> GeLU
    let c2_w = b.add_input("conv2_w", &[D_MODEL, D_MODEL, CONV_KERNEL]);
    let c2_b = b.add_input("conv2_b", &[D_MODEL]);
    let c2 = b.add_conv1d(g1, c2_w, Some(c2_b), 2, CONV_PADDING, &[D_MODEL, t_out]);
    let out = b.add_gelu(c2, &[D_MODEL, t_out]);

    (b.build(out).expect("valid mel+GeLU kernel"), t_out)
}

fn mel_features_gelu_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, N_MEL, CONV_KERNEL]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[D_MODEL, D_MODEL, CONV_KERNEL]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
    ]
}

#[test]
fn test_whisper_mel_gelu_crown_after_fix() {
    let (def, t_out) = build_mel_features_gelu();
    let bindings = mel_features_gelu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_eq!(ibp_output.lower_upper().0.shape(), &[D_MODEL, t_out]);
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("Whisper mel+GeLU IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    // CROWN -- tests the fixed CROWN relaxation through conv+GeLU chains
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(crown_output.lower_upper().0.shape(), &[D_MODEL, t_out]);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "Whisper mel+GeLU CROWN: method={method:?}, [{crown_lo}, {crown_hi}], width={crown_width}"
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    // With small weights, bounds should be moderate
    assert!(ibp_lo.abs() < 100.0, "IBP lower < 100, got {ibp_lo}");
    assert!(ibp_hi.abs() < 100.0, "IBP upper < 100, got {ibp_hi}");
}

// ===========================================================================
// 6. Recording tests -- write CROWN reverify results to status file
// ===========================================================================
//
// These tests use `verify_and_assert` / `verify_and_assert_with_config` to
// record CROWN verification results to `nn_verify_status_whisper.json`.
// The pipeline's `run_escalation` automatically attempts CROWN after IBP.

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

/// Record isolated GeLU CROWN verification to status file.
///
/// Establishes a CROWN baseline for standalone GeLU activation after the
/// NY GeLU relaxation fix (e810fb2b). Whisper uses gelu_erf
/// extensively in encoder and decoder FFN blocks.
#[test]
fn test_whisper_gelu_isolated_record_crown_reverify() {
    let (def, t) = build_gelu_isolated();
    let bindings = vec![TensorParamBinding::Variable];
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "whisper_gelu_isolated_crown_reverify",
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD whisper_gelu_isolated_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}

/// Record encoder FFN+GeLU CROWN verification to status file.
///
/// Tests the encoder FFN pattern:
/// LayerNorm -> Linear(D, 4D) -> GeLU -> Linear(4D, D) -> residual.
/// Uses Conservative NormBoundsMode since LayerNorm is present.
#[test]
fn test_whisper_enc_ffn_gelu_record_crown_reverify() {
    let (def, t) = build_encoder_ffn_gelu();
    let bindings = encoder_ffn_gelu_bindings();
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "whisper_enc_ffn_gelu_crown_reverify",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD whisper_enc_ffn_gelu_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}

/// Record full encoder block (self-attention + FFN with GeLU) CROWN verification.
///
/// Tests the complete encoder block: LayerNorm -> MHA -> residual ->
/// LayerNorm -> Linear -> GeLU -> Linear -> residual.
/// Uses Conservative NormBoundsMode.
#[test]
fn test_whisper_enc_block_gelu_record_crown_reverify() {
    let (def, t) = build_encoder_block_with_gelu();
    let bindings = encoder_block_gelu_bindings();
    let input = uniform_bounds(&[t, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "whisper_enc_block_gelu_crown_reverify",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD whisper_enc_block_gelu_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}

/// Record mel feature extraction (Conv1d -> GeLU chain) CROWN verification.
///
/// Tests the conv stem: Conv1d(s=1) -> GeLU -> Conv1d(s=2) -> GeLU.
/// No normalization layers, so default config is used.
#[test]
fn test_whisper_mel_gelu_record_crown_reverify() {
    let (def, _) = build_mel_features_gelu();
    let bindings = mel_features_gelu_bindings();
    let input = uniform_bounds(&[N_MEL, MEL_SEQ_LEN], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "whisper_mel_gelu_crown_reverify");

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD whisper_mel_gelu_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}
