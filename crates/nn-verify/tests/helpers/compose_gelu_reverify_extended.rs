// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended GeLU re-verification compose tests across model pipelines after
//! NY CROWN relaxation soundness fix (rev e810fb2b).
//!
//! Existing GeLU re-verification covers Whisper (encoder FFN, mel stem) and
//! HTDemucs (temporal encoder, DConv, transformer FFN). This module extends
//! coverage to additional architectures:
//!
//! 1. **Standalone GeLU CROWN bounds**: Single GeLU layer, tight CROWN bounds.
//!
//! 2. **GeLU in FFN context**: Linear -> GeLU -> Linear, verify CROWN through
//!    the standard FFN pattern used in Qwen3 decoder and GLM5.
//!
//! 3. **GeLU vs SwiGLU comparison**: Build both activation FFN patterns,
//!    compare bound tightness (GeLU FFN vs SwiGLU FFN).
//!
//! 4. **GeLU with LayerNorm pre-activation**: LayerNorm -> Linear -> GeLU ->
//!    Linear pattern used in standard transformer encoder FFNs.
//!
//! 5. **GeLU quantization sensitivity**: Test that bounds hold when weight
//!    magnitudes approximate bf16/f16 quantization scenarios.
//!
//! 6. **Multi-layer GeLU stack**: 3-layer MLP with GeLU activations, verify
//!    bounds don't blow up through depth.
//!
//! 7. **GeLU with residual connection**: Test GeLU in a residual block
//!    (x + GeLU(Linear(x))), as used in Qwen3/GLM5 decoder layers.
//!
//! All tests use IbpValidated soundness mode via Conservative NormBoundsMode
//! where normalization layers are present.
//!
//! Part of #4314: Re-verify GeLU models after NY CROWN relaxation fix.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert, verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding, VerifyConfig};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Model / hidden dimension (Qwen3/GLM5 style decoder).
const D_MODEL: usize = 32;
/// FFN intermediate dimension (2x hidden for test speed).
const FFN_DIM: usize = 64;
/// Sequence length.
const SEQ_LEN: usize = 4;
/// Small weight magnitude for bounded verification.
const W_MAG: f32 = 0.02;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// ===========================================================================
// Helper: SiLU activation (for SwiGLU comparison)
// ===========================================================================

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
// 1. Standalone GeLU CROWN bounds
// ===========================================================================

/// Build standalone GeLU activation on a 2D tensor.
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
fn build_gelu_standalone() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("gelu_reverify_standalone");
    let x = b.add_input("x", &[SEQ_LEN, D_MODEL]);
    let out = b.add_gelu(x, &[SEQ_LEN, D_MODEL]);
    b.build(out).expect("valid standalone GeLU kernel")
}

#[test]
fn test_gelu_standalone_crown_bounds() {
    let def = build_gelu_standalone();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    // IBP baseline
    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("GeLU standalone IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    // CROWN with fixed relaxation
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "GeLU standalone CROWN: method={method:?}, [{crown_lo}, {crown_hi}], width={crown_width}"
    );
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

#[test]
fn test_gelu_standalone_tight_input_crown() {
    let def = build_gelu_standalone();
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Tight input bounds +-0.1 for higher CROWN precision
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 0.1);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;

    eprintln!(
        "GeLU standalone tight IBP width={ibp_width:.6}, CROWN width={crown_width:.6} \
         (method={method:?})"
    );

    assert!(ibp_lo.is_finite() && ibp_hi.is_finite());
    assert!(crown_lo.is_finite() && crown_hi.is_finite());
}

/// Record standalone GeLU CROWN verification to status file.
#[test]
fn test_gelu_standalone_record_crown_reverify() {
    let def = build_gelu_standalone();
    let bindings = vec![TensorParamBinding::Variable];
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "gelu_standalone_crown_reverify");

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD gelu_standalone_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}

// ===========================================================================
// 2. GeLU in FFN context (Qwen3/GLM5 decoder pattern)
// ===========================================================================

/// Build FFN: Linear(D, FFN_DIM) -> GeLU -> Linear(FFN_DIM, D).
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
fn build_gelu_ffn() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gelu_reverify_ffn");

    let x = b.add_input("x", &[SEQ_LEN, D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    let h = b.add_linear(x, ffn1_w, None, &[SEQ_LEN, FFN_DIM]);
    let act = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);
    let out = b.add_linear(act, ffn2_w, None, &[SEQ_LEN, D_MODEL]);

    let def = b.build(out).expect("valid GeLU FFN kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, FFN_DIM]), W_MAG)),
    ];
    (def, bindings)
}

#[test]
fn test_gelu_ffn_crown_bounds() {
    let (def, bindings) = build_gelu_ffn();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("GeLU FFN IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!("GeLU FFN CROWN: method={method:?}, [{crown_lo}, {crown_hi}], width={crown_width}");
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    assert!(ibp_lo.is_finite() && ibp_hi.is_finite());
    assert!(crown_lo.is_finite() && crown_hi.is_finite());
}

/// Record GeLU FFN CROWN verification to status file.
#[test]
fn test_gelu_ffn_record_crown_reverify() {
    let (def, bindings) = build_gelu_ffn();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "gelu_ffn_crown_reverify");

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD gelu_ffn_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}

// ===========================================================================
// 3. GeLU vs SwiGLU comparison
// ===========================================================================

/// Build SwiGLU FFN: gate_proj -> SiLU -> mul(up_proj) -> down_proj.
///
/// Same input/output shape as the GeLU FFN for direct comparison.
fn build_swiglu_ffn() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gelu_reverify_swiglu_compare");

    let x = b.add_input("x", &[SEQ_LEN, D_MODEL]);
    let gate_w = b.add_input("gate_w", &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input("up_w", &[FFN_DIM, D_MODEL]);
    let down_w = b.add_input("down_w", &[D_MODEL, FFN_DIM]);

    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, D_MODEL];

    // gate_proj -> SiLU
    let gate = b.add_linear(x, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);

    // up_proj
    let up = b.add_linear(x, up_w, None, &ffn_shape);

    // element-wise gate * up -> down_proj
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &out_shape);

    let def = b.build(out).expect("valid SwiGLU FFN kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, FFN_DIM]), W_MAG)),
    ];
    (def, bindings)
}

/// Compare GeLU FFN vs SwiGLU FFN bound tightness.
///
/// Both patterns are standard in modern LLMs. GeLU is used in GPT-2/BERT/Whisper;
/// SwiGLU is used in Llama/Qwen3/GLM5. Comparing bound widths helps understand
/// the relative verification difficulty of each activation pattern.
#[test]
fn test_gelu_vs_swiglu_bound_comparison() {
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    // GeLU FFN
    let (gelu_def, gelu_bindings) = build_gelu_ffn();
    let gelu_graph = tensor_kernel_to_graph(&gelu_def, &gelu_bindings).expect("gelu graph");
    let gelu_ibp = gelu_graph.propagate_ibp(&input).expect("GeLU IBP");
    let (gelu_lo, gelu_hi) = bounds_min_max(&gelu_ibp);
    let gelu_width = gelu_hi - gelu_lo;

    // SwiGLU FFN
    let (swiglu_def, swiglu_bindings) = build_swiglu_ffn();
    let swiglu_graph = tensor_kernel_to_graph(&swiglu_def, &swiglu_bindings).expect("swiglu graph");
    let swiglu_ibp = swiglu_graph.propagate_ibp(&input).expect("SwiGLU IBP");
    let (swiglu_lo, swiglu_hi) = bounds_min_max(&swiglu_ibp);
    let swiglu_width = swiglu_hi - swiglu_lo;

    eprintln!(
        "GeLU FFN IBP width={gelu_width:.6}, SwiGLU FFN IBP width={swiglu_width:.6}, \
         ratio={:.2}x",
        if swiglu_width > 0.0 {
            gelu_width / swiglu_width
        } else {
            f32::INFINITY
        }
    );

    // Both must produce finite, valid bounds
    assert!(gelu_lo.is_finite() && gelu_hi.is_finite());
    assert!(swiglu_lo.is_finite() && swiglu_hi.is_finite());
    assert!(gelu_width >= 0.0, "GeLU width must be non-negative");
    assert!(swiglu_width >= 0.0, "SwiGLU width must be non-negative");
}

/// Compare GeLU vs SwiGLU CROWN tightness.
#[test]
fn test_gelu_vs_swiglu_crown_comparison() {
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    // GeLU FFN CROWN
    let (gelu_def, gelu_bindings) = build_gelu_ffn();
    let gelu_graph = tensor_kernel_to_graph(&gelu_def, &gelu_bindings).expect("gelu graph");
    let (gelu_method, gelu_crown, _) = assert_crown_tighter_when_not_fallback(&gelu_graph, &input);
    let (gelu_lo, gelu_hi) = bounds_min_max(&gelu_crown);
    let gelu_width = gelu_hi - gelu_lo;

    // SwiGLU FFN CROWN
    let (swiglu_def, swiglu_bindings) = build_swiglu_ffn();
    let swiglu_graph = tensor_kernel_to_graph(&swiglu_def, &swiglu_bindings).expect("swiglu graph");
    let (swiglu_method, swiglu_crown, _) =
        assert_crown_tighter_when_not_fallback(&swiglu_graph, &input);
    let (swiglu_lo, swiglu_hi) = bounds_min_max(&swiglu_crown);
    let swiglu_width = swiglu_hi - swiglu_lo;

    eprintln!("GeLU FFN CROWN: method={gelu_method:?}, width={gelu_width:.6}");
    eprintln!("SwiGLU FFN CROWN: method={swiglu_method:?}, width={swiglu_width:.6}");

    assert!(gelu_lo.is_finite() && gelu_hi.is_finite());
    assert!(swiglu_lo.is_finite() && swiglu_hi.is_finite());
}

// ===========================================================================
// 4. GeLU with LayerNorm pre-activation
// ===========================================================================

/// Build LayerNorm -> Linear -> GeLU -> Linear (pre-norm FFN pattern).
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
fn build_layernorm_gelu_ffn() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gelu_reverify_ln_ffn");

    let x = b.add_input("x", &[SEQ_LEN, D_MODEL]);
    let eps = b.add_input("eps", &[1]);
    let ln_w = b.add_input("ln_w", &[D_MODEL]);
    let ln_b = b.add_input("ln_b", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    let shape = [SEQ_LEN, D_MODEL];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Pre-norm: LayerNorm
    let normed = b.add_layer_norm(x, eps, 1, ln_w, ln_b, &shape);
    // FFN: Linear(D, FFN_DIM) -> GeLU -> Linear(FFN_DIM, D)
    let h = b.add_linear(normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(h, &ffn_shape);
    let out = b.add_linear(act, ffn2_w, None, &shape);

    let def = b.build(out).expect("valid LayerNorm+GeLU FFN kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, FFN_DIM]), W_MAG)),
    ];
    (def, bindings)
}

#[test]
fn test_layernorm_gelu_ffn_ibp() {
    let (def, bindings) = build_layernorm_gelu_ffn();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("LayerNorm+GeLU FFN IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    assert!(ibp_lo.is_finite() && ibp_hi.is_finite());
}

#[test]
fn test_layernorm_gelu_ffn_crown() {
    let (def, bindings) = build_layernorm_gelu_ffn();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "LayerNorm+GeLU FFN CROWN: method={method:?}, [{crown_lo}, {crown_hi}], \
         width={crown_width}"
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    assert!(crown_lo.is_finite() && crown_hi.is_finite());
}

/// Record LayerNorm+GeLU FFN CROWN verification to status file.
#[test]
fn test_layernorm_gelu_ffn_record_crown_reverify() {
    let (def, bindings) = build_layernorm_gelu_ffn();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "gelu_layernorm_ffn_crown_reverify",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD gelu_layernorm_ffn_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}

// ===========================================================================
// 5. GeLU quantization sensitivity
// ===========================================================================

/// Test GeLU FFN bounds stability with different weight magnitudes
/// approximating quantization scenarios.
///
/// bf16 has ~3 digits of precision (epsilon ~0.00781), f16 has ~3.3 digits
/// (epsilon ~0.000977). We test that CROWN bounds remain finite and reasonable
/// at different weight scales that approximate quantized weight distributions.
#[test]
fn test_gelu_quantization_sensitivity_ibp() {
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    // Test at different weight magnitudes: small (fp32-like), medium (bf16-like), larger
    for (label, w_mag) in [
        ("fp32_small", 0.01_f32),
        ("bf16_typical", 0.05),
        ("f16_typical", 0.1),
    ] {
        let mut b = TensorBlockBuilder::new(&format!("gelu_reverify_quant_{label}"));
        let x = b.add_input("x", &[SEQ_LEN, D_MODEL]);
        let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
        let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

        let h = b.add_linear(x, ffn1_w, None, &[SEQ_LEN, FFN_DIM]);
        let act = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);
        let out = b.add_linear(act, ffn2_w, None, &[SEQ_LEN, D_MODEL]);

        let def = b.build(out).expect("valid quantization GeLU kernel");
        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[FFN_DIM, D_MODEL]),
                w_mag,
            )),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[D_MODEL, FFN_DIM]),
                w_mag,
            )),
        ];

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let ibp_output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&ibp_output);
        let (lo, hi) = bounds_min_max(&ibp_output);
        let width = hi - lo;
        eprintln!("GeLU quant {label} (w_mag={w_mag}): IBP [{lo:.6}, {hi:.6}], width={width:.6}");

        assert!(lo.is_finite(), "bounds must be finite for {label}");
        assert!(hi.is_finite(), "bounds must be finite for {label}");
        assert!(width >= 0.0, "width must be non-negative for {label}");
    }
}

/// Test that CROWN bounds remain sound across quantization-like weight scales.
#[test]
fn test_gelu_quantization_sensitivity_crown() {
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    for (label, w_mag) in [("small", 0.01_f32), ("medium", 0.05), ("large", 0.1)] {
        let mut b = TensorBlockBuilder::new(&format!("gelu_reverify_quant_crown_{label}"));
        let x = b.add_input("x", &[SEQ_LEN, D_MODEL]);
        let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
        let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

        let h = b.add_linear(x, ffn1_w, None, &[SEQ_LEN, FFN_DIM]);
        let act = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);
        let out = b.add_linear(act, ffn2_w, None, &[SEQ_LEN, D_MODEL]);

        let def = b.build(out).expect("valid quantization GeLU kernel");
        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[FFN_DIM, D_MODEL]),
                w_mag,
            )),
            TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[D_MODEL, FFN_DIM]),
                w_mag,
            )),
        ];

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let (method, crown_output, _) = assert_crown_tighter_when_not_fallback(&graph, &input);
        let (lo, hi) = bounds_min_max(&crown_output);
        let width = hi - lo;
        eprintln!("GeLU quant CROWN {label} (w_mag={w_mag}): method={method:?}, width={width:.6}");

        assert!(lo.is_finite() && hi.is_finite());
    }
}

// ===========================================================================
// 6. Multi-layer GeLU stack (3-layer MLP)
// ===========================================================================

/// Build 3-layer MLP: Linear -> GeLU -> Linear -> GeLU -> Linear -> GeLU -> Linear.
///
/// Tests that CROWN bounds don't blow up through depth.
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
fn build_multi_layer_gelu_mlp() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gelu_reverify_multi_layer");

    let x = b.add_input("x", &[SEQ_LEN, D_MODEL]);
    let w1 = b.add_input("w1", &[FFN_DIM, D_MODEL]);
    let w2 = b.add_input("w2", &[FFN_DIM, FFN_DIM]);
    let w3 = b.add_input("w3", &[FFN_DIM, FFN_DIM]);
    let w_out = b.add_input("w_out", &[D_MODEL, FFN_DIM]);

    let shape = [SEQ_LEN, D_MODEL];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // Layer 1: Linear -> GeLU
    let h1 = b.add_linear(x, w1, None, &ffn_shape);
    let a1 = b.add_gelu(h1, &ffn_shape);

    // Layer 2: Linear -> GeLU
    let h2 = b.add_linear(a1, w2, None, &ffn_shape);
    let a2 = b.add_gelu(h2, &ffn_shape);

    // Layer 3: Linear -> GeLU
    let h3 = b.add_linear(a2, w3, None, &ffn_shape);
    let a3 = b.add_gelu(h3, &ffn_shape);

    // Output projection
    let out = b.add_linear(a3, w_out, None, &shape);

    let def = b.build(out).expect("valid multi-layer GeLU MLP kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, FFN_DIM]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, FFN_DIM]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, FFN_DIM]), W_MAG)),
    ];
    (def, bindings)
}

#[test]
fn test_multi_layer_gelu_mlp_ibp_depth_widening() {
    let (def, bindings) = build_multi_layer_gelu_mlp();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("3-layer GeLU MLP IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    // Bounds should be finite even through 3 layers with small weights
    assert!(ibp_lo.is_finite(), "3-layer IBP lower must be finite");
    assert!(ibp_hi.is_finite(), "3-layer IBP upper must be finite");
}

#[test]
fn test_multi_layer_gelu_mlp_crown() {
    let (def, bindings) = build_multi_layer_gelu_mlp();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "3-layer GeLU MLP CROWN: method={method:?}, [{crown_lo}, {crown_hi}], \
         width={crown_width}"
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    assert!(crown_lo.is_finite() && crown_hi.is_finite());
}

/// Record 3-layer GeLU MLP CROWN verification to status file.
#[test]
fn test_multi_layer_gelu_mlp_record_crown_reverify() {
    let (def, bindings) = build_multi_layer_gelu_mlp();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "gelu_multi_layer_mlp_crown_reverify",
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD gelu_multi_layer_mlp_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}

/// Compare 1-layer vs 3-layer GeLU MLP IBP width to quantify depth widening.
#[test]
fn test_gelu_depth_widening_analysis() {
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    // 1-layer: Linear -> GeLU -> Linear
    let (def_1, bindings_1) = build_gelu_ffn();
    let graph_1 = tensor_kernel_to_graph(&def_1, &bindings_1).expect("1-layer graph");
    let ibp_1 = graph_1.propagate_ibp(&input).expect("1-layer IBP");
    let (lo_1, hi_1) = bounds_min_max(&ibp_1);
    let width_1 = hi_1 - lo_1;

    // 3-layer: Linear -> GeLU -> Linear -> GeLU -> Linear -> GeLU -> Linear
    let (def_3, bindings_3) = build_multi_layer_gelu_mlp();
    let graph_3 = tensor_kernel_to_graph(&def_3, &bindings_3).expect("3-layer graph");
    let ibp_3 = graph_3.propagate_ibp(&input).expect("3-layer IBP");
    let (lo_3, hi_3) = bounds_min_max(&ibp_3);
    let width_3 = hi_3 - lo_3;

    eprintln!(
        "Depth widening: 1-layer width={width_1:.6}, 3-layer width={width_3:.6}, \
         ratio={:.2}x",
        if width_1 > 0.0 {
            width_3 / width_1
        } else {
            f32::INFINITY
        }
    );

    assert!(width_1.is_finite() && width_3.is_finite());
    // 3-layer bounds should be wider but still finite with small weights
    assert!(width_3 >= 0.0);
}

// ===========================================================================
// 7. GeLU with residual connection
// ===========================================================================

/// Build residual block: x + Linear(GeLU(Linear(x))).
///
/// This is the standard FFN residual pattern in Qwen3/GLM5 decoders:
/// residual = x + down_proj(GeLU(up_proj(x))).
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
fn build_gelu_residual() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gelu_reverify_residual");

    let x = b.add_input("x", &[SEQ_LEN, D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    let shape = [SEQ_LEN, D_MODEL];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // FFN branch: Linear -> GeLU -> Linear
    let h = b.add_linear(x, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(h, &ffn_shape);
    let proj = b.add_linear(act, ffn2_w, None, &shape);

    // Residual: x + FFN(x)
    let out = b.add_binary_add(x, proj, &shape);

    let def = b.build(out).expect("valid GeLU residual kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, FFN_DIM]), W_MAG)),
    ];
    (def, bindings)
}

#[test]
fn test_gelu_residual_ibp() {
    let (def, bindings) = build_gelu_residual();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("GeLU residual IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    // Residual should widen bounds by the FFN contribution
    // But with small weights, the widening should be modest
    assert!(ibp_lo.is_finite() && ibp_hi.is_finite());
    assert!(
        ibp_width >= 2.0,
        "residual must be at least as wide as input"
    );
}

#[test]
fn test_gelu_residual_crown() {
    let (def, bindings) = build_gelu_residual();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "GeLU residual CROWN: method={method:?}, [{crown_lo}, {crown_hi}], width={crown_width}"
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    assert!(crown_lo.is_finite() && crown_hi.is_finite());
}

/// Record GeLU residual CROWN verification to status file.
#[test]
fn test_gelu_residual_record_crown_reverify() {
    let (def, bindings) = build_gelu_residual();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "gelu_residual_crown_reverify");

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD gelu_residual_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}

// ===========================================================================
// 8. RMSNorm + GeLU FFN (Qwen3/GLM5 decoder pattern)
// ===========================================================================

/// Build RMSNorm -> Linear -> GeLU -> Linear (Qwen3/GLM5 decoder FFN).
///
/// Input: `[SEQ_LEN, D_MODEL]` (Variable).
/// Output: `[SEQ_LEN, D_MODEL]`.
fn build_rmsnorm_gelu_ffn() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let mut b = TensorBlockBuilder::new("gelu_reverify_rmsnorm_ffn");

    let x = b.add_input("x", &[SEQ_LEN, D_MODEL]);
    let rms_eps = b.add_input("rms_eps", &[1]);
    let rms_w = b.add_input("rms_w", &[D_MODEL]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, D_MODEL]);
    let ffn2_w = b.add_input("ffn2_w", &[D_MODEL, FFN_DIM]);

    let shape = [SEQ_LEN, D_MODEL];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    // RMSNorm pre-activation
    let normed = b.add_rms_norm(x, rms_eps, 1, rms_w, &shape);
    // FFN: Linear -> GeLU -> Linear
    let h = b.add_linear(normed, ffn1_w, None, &ffn_shape);
    let act = b.add_gelu(h, &ffn_shape);
    let out = b.add_linear(act, ffn2_w, None, &shape);

    let def = b.build(out).expect("valid RMSNorm+GeLU FFN kernel");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[FFN_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, FFN_DIM]), W_MAG)),
    ];
    (def, bindings)
}

#[test]
fn test_rmsnorm_gelu_ffn_ibp() {
    let (def, bindings) = build_rmsnorm_gelu_ffn();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!("RMSNorm+GeLU FFN IBP: [{ibp_lo}, {ibp_hi}], width={ibp_width}");

    assert!(ibp_lo.is_finite() && ibp_hi.is_finite());
}

#[test]
fn test_rmsnorm_gelu_ffn_crown() {
    let (def, bindings) = build_rmsnorm_gelu_ffn();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;
    eprintln!(
        "RMSNorm+GeLU FFN CROWN: method={method:?}, [{crown_lo}, {crown_hi}], \
         width={crown_width}"
    );
    if let Some(r) = &fallback_reason {
        eprintln!("  fallback: {r}");
    }

    assert!(crown_lo.is_finite() && crown_hi.is_finite());
}

/// Record RMSNorm+GeLU FFN CROWN verification to status file.
#[test]
fn test_rmsnorm_gelu_ffn_record_crown_reverify() {
    let (def, bindings) = build_rmsnorm_gelu_ffn();
    let input = uniform_bounds(&[SEQ_LEN, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "gelu_rmsnorm_ffn_crown_reverify",
        &conservative_config(),
    );

    let (lo, hi) = bounds_min_max(&result.output_bounds);
    let width = hi - lo;
    eprintln!(
        "RECORD gelu_rmsnorm_ffn_crown_reverify: [{lo}, {hi}], width={width}, \
         method={:?}, soundness={:?}",
        result.verification.method, result.verification.soundness_mode
    );
}

// ===========================================================================
// 9. IBP vs CROWN tightness sweep across input ranges
// ===========================================================================

/// Compare IBP and CROWN bounds width for GeLU FFN at different input ranges.
/// After the CROWN relaxation fix, CROWN should produce tighter bounds than
/// IBP for GeLU-containing blocks.
#[test]
fn test_gelu_ibp_vs_crown_tightness_sweep() {
    let (def, bindings) = build_gelu_ffn();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    for range in [0.1_f32, 0.5, 1.0, 2.0, 5.0] {
        let input = uniform_bounds(&[SEQ_LEN, D_MODEL], range);

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

        assert!(ibp_lo.is_finite() && ibp_hi.is_finite());
        assert!(crown_lo.is_finite() && crown_hi.is_finite());
        assert!(ibp_width >= 0.0, "IBP width must be non-negative");
        assert!(crown_width >= 0.0, "CROWN width must be non-negative");
    }
}
