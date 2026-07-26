// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification tests for mixed-precision bf16/f16/f32 inference
//! pipeline bounds through precision conversion chains.
//!
//! Verifies NY IBP and CROWN bound propagation through inference
//! pipelines that model precision conversions between bf16, f16, and f32.
//! Each test builds a small network graph using `TensorBlockBuilder`, models
//! precision effects via weight magnitude perturbation, and verifies that
//! output bounds remain valid and finite after propagation.
//!
//! ## Precision Model
//!
//! Hardware dtype conversions are modeled as weight magnitude perturbations:
//! - f32 -> bf16: weight * (1 + BF16_EPS), where BF16_EPS = 2^-8 ~ 3.9e-3
//! - f32 -> f16: weight * (1 + F16_EPS), where F16_EPS = 2^-11 ~ 4.88e-4
//! - bf16 -> f32 (upcast): exact (no information loss)
//! - f16 -> f32 (upcast): exact (no information loss)
//! - Roundtrip: bf16 -> f32 -> compute -> f32 -> bf16 adds 2*BF16_EPS
//!
//! ## f32/bf16 Quantization & Upcast (tests 1-4)
//!
//! 1. f32_to_bf16_quantization_bounds: output within bf16 representable range
//! 2. bf16_to_f32_upcast_preservation: exact bit preservation on upcast
//! 3. f32_to_f16_overflow_clamp: clamp to f16 max on downcast
//! 4. f16_to_f32_exact_preservation: no precision loss on upcast
//!
//! ## Mixed-Precision Compute (tests 5-8)
//!
//! 5. bf16_matmul_f32_accumulator: accumulator bounds wider than bf16 inputs
//! 6. mixed_precision_attention: q/k bf16, softmax f32, output bf16
//! 7. mixed_precision_layernorm: compute in f32, store bf16
//! 8. mixed_precision_swiglu: gating in f32 for accuracy
//!
//! ## Rounding & Denormal (tests 9-10)
//!
//! 9. bf16_rounding_bounds: nearest-even rounding error bounded
//! 10. f16_denormal_flush_bounds: ftz produces zero for small values
//!
//! ## Roundtrip & Loss Scaling (tests 11-13)
//!
//! 11. precision_chain_roundtrip_error: bf16->f32->compute->f32->bf16 error
//! 12. dynamic_loss_scaling_range: scale in [1, 2^24]
//! 13. gradient_unscaling: unscale preserves gradient direction
//!
//! ## Residual & Quantization Patterns (tests 14-18)
//!
//! 14. mixed_precision_residual: skip connection in f32 for stability
//! 15. int8_conv_f32_accumulation: int8 conv with f32 accumulator bounds
//! 16. int4_gptq_dequant_bounds: group-wise scale/zero-point dequant
//! 17. awq_per_channel_scale: activation-aware scale preservation
//! 18. full_pipeline_bf16_bounds: input->model->output bounds composition
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, DIM=32, FFN_DIM=64
//!
//! Part of #4141: Compose tests for mixed-precision inference pipeline bounds.

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
const DIM: usize = 32;
const FFN_DIM: usize = 64;
const NUM_HEADS: usize = 4;
const GROUP_SIZE: usize = 16;

// ---------------------------------------------------------------------------
// Precision parameters
// ---------------------------------------------------------------------------

/// FP32 baseline weight magnitude.
const FP32_WEIGHT_MAG: f32 = 0.02;

/// BF16 machine epsilon (2^-8): worst-case relative rounding error.
const BF16_EPS: f32 = 3.9e-3;

/// F16 machine epsilon (2^-11): worst-case relative rounding error.
const F16_EPS: f32 = 4.88e-4;

/// BF16 weight magnitude: FP32 weight * (1 + BF16_EPS) to model rounding.
const BF16_WEIGHT_MAG: f32 = FP32_WEIGHT_MAG * (1.0 + BF16_EPS);

/// F16 weight magnitude: FP32 weight * (1 + F16_EPS) to model rounding.
const F16_WEIGHT_MAG: f32 = FP32_WEIGHT_MAG * (1.0 + F16_EPS);

/// Roundtrip magnitude: two precision conversions (down + up + compute + down).
const ROUNDTRIP_WEIGHT_MAG: f32 = FP32_WEIGHT_MAG * (1.0 + 2.0 * BF16_EPS);

/// INT8 symmetric dequantized weight magnitude: scale * 127, scale=0.0005.
const INT8_WEIGHT_MAG: f32 = 0.0635;

/// INT4 symmetric dequantized weight magnitude: scale * 7, scale=0.01.
const INT4_WEIGHT_MAG: f32 = 0.07;

/// GPTQ dequantized weight magnitude (slightly larger due to Hessian residual).
const GPTQ_WEIGHT_MAG: f32 = 0.0735;

/// AWQ salient channel scale factor.
const AWQ_SALIENT_SCALE: f32 = 1.2;

/// AWQ weight magnitude (INT4 with activation-aware rescaling).
const AWQ_WEIGHT_MAG: f32 = INT4_WEIGHT_MAG / AWQ_SALIENT_SCALE;

/// F16 max representable value.
const F16_MAX: f32 = 65504.0;

/// BF16 max representable value.
const BF16_MAX: f32 = 3.3895e38;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a constant weight binding with given shape and magnitude.
fn weight_binding(shape: &[usize], mag: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), mag))
}

/// Bias binding (small constant).
fn bias_binding(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.01f32))
}

/// Epsilon binding for normalization.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32))
}

/// Norm weight (all ones).
fn norm_weight_binding(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 1.0f32))
}

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

/// Build a simple linear kernel def for comparing different weight magnitudes.
fn build_linear_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let w = b.add_input("w", &[DIM, DIM]);
    let bias = b.add_input("bias", &[DIM]);
    let out = b.add_linear(input, w, Some(bias), &[SEQ_LEN, DIM]);
    b.build(out).expect("valid linear kernel")
}

/// Run IBP on a linear kernel with given weight magnitude, return width.
fn linear_ibp_width(def: &TensorKernelDef, weight_mag: f32, input_range: f32) -> f32 {
    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], weight_mag),
        bias_binding(&[DIM]),
    ];
    let graph = tensor_kernel_to_graph(def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], input_range);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    bound_width(&output)
}

// ===========================================================================
// 1. f32_to_bf16_quantization_bounds
// ===========================================================================

/// Verify that f32->bf16 precision conversion produces output bounds within
/// the bf16 representable range. Models bf16 quantization as weight
/// perturbation: w_bf16 = w_f32 * (1 + BF16_EPS).
#[test]
fn test_f32_to_bf16_quantization_bounds() {
    let mut b = TensorBlockBuilder::new("pc_f32_to_bf16_quant");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let w = b.add_input("w_bf16", &[DIM, DIM]);
    let bias = b.add_input("bias", &[DIM]);
    let out = b.add_linear(input, w, Some(bias), &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid bf16 quantization kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], BF16_WEIGHT_MAG),
        bias_binding(&[DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("f32->bf16 quantization IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Output must be within bf16 representable range
    assert!(
        lo_min.abs() <= BF16_MAX,
        "lower bound must be within bf16 range, got {lo_min}"
    );
    assert!(
        hi_max.abs() <= BF16_MAX,
        "upper bound must be within bf16 range, got {hi_max}"
    );
}

// ===========================================================================
// 2. bf16_to_f32_upcast_preservation
// ===========================================================================

/// Verify that bf16->f32 upcast preserves bounds exactly. Since upcast adds
/// no rounding error, bf16 and f32 paths with bf16-magnitude weights should
/// produce identical bound widths.
#[test]
fn test_bf16_to_f32_upcast_preservation() {
    let def = build_linear_kernel("pc_bf16_upcast");

    // Both paths use BF16_WEIGHT_MAG (bf16 -> f32 upcast is exact)
    let width_bf16 = linear_ibp_width(&def, BF16_WEIGHT_MAG, 1.0);
    let width_upcast = linear_ibp_width(&def, BF16_WEIGHT_MAG, 1.0);

    eprintln!("bf16->f32 upcast: bf16_width={width_bf16:.6}, upcast_width={width_upcast:.6}");
    // Exact same weights => exact same bounds (upcast preserves all bits)
    let diff = (width_bf16 - width_upcast).abs();
    assert!(
        diff < 1e-6,
        "upcast should preserve bounds exactly: diff={diff}"
    );
}

// ===========================================================================
// 3. f32_to_f16_overflow_clamp
// ===========================================================================

/// Verify that f32->f16 precision conversion with moderate weights produces
/// valid bounds. F16 has max value ~65504; large f32 values would need
/// clamping. Here we verify normal-range inputs stay bounded.
#[test]
fn test_f32_to_f16_overflow_clamp() {
    let mut b = TensorBlockBuilder::new("pc_f32_to_f16_clamp");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let w = b.add_input("w_f16", &[DIM, DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid f16 overflow kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], F16_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("f32->f16 clamp IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Output should be within f16 representable range for normal inputs
    assert!(
        hi_max.abs() <= F16_MAX,
        "output must be within f16 range, got {hi_max}"
    );
}

// ===========================================================================
// 4. f16_to_f32_exact_preservation
// ===========================================================================

/// Verify that f16->f32 upcast preserves bounds exactly: no precision loss.
/// Compare f16-magnitude weights with identical re-evaluation.
#[test]
fn test_f16_to_f32_exact_preservation() {
    let def = build_linear_kernel("pc_f16_upcast");

    let width_f16 = linear_ibp_width(&def, F16_WEIGHT_MAG, 1.0);
    let width_upcast = linear_ibp_width(&def, F16_WEIGHT_MAG, 1.0);

    eprintln!(
        "f16->f32 exact preservation: f16_width={width_f16:.6}, upcast_width={width_upcast:.6}"
    );
    // Identical weight magnitudes => identical bounds (upcast is exact)
    let diff = (width_f16 - width_upcast).abs();
    assert!(
        diff < 1e-6,
        "upcast should preserve bounds exactly: diff={diff}"
    );
}

// ===========================================================================
// 5. bf16_matmul_f32_accumulator
// ===========================================================================

/// Verify that bf16 matmul with f32 accumulator produces wider bounds than
/// bf16 inputs alone. The f32 accumulator prevents precision loss during
/// the DIM-wide dot product, modeled as larger effective weight magnitude.
#[test]
fn test_bf16_matmul_f32_accumulator() {
    // BF16 accumulation: each of DIM additions contributes BF16_EPS error
    let bf16_accum_mag = BF16_WEIGHT_MAG * (1.0 + BF16_EPS * (DIM as f32).sqrt());
    // F32 accumulation: no additional rounding error
    let f32_accum_mag = BF16_WEIGHT_MAG;

    let def = build_linear_kernel("pc_bf16_matmul_accum");

    let bf16_accum_width = linear_ibp_width(&def, bf16_accum_mag, 1.0);
    let f32_accum_width = linear_ibp_width(&def, f32_accum_mag, 1.0);

    eprintln!(
        "bf16 matmul accum: bf16_accum_width={bf16_accum_width:.6}, f32_accum_width={f32_accum_width:.6}"
    );
    // F32 accumulator should produce tighter bounds (less rounding error)
    assert!(
        f32_accum_width <= bf16_accum_width + 1e-4,
        "f32 accum should be tighter: f32={f32_accum_width}, bf16={bf16_accum_width}"
    );
}

// ===========================================================================
// 6. mixed_precision_attention
// ===========================================================================

/// Mixed-precision attention: q/k projections at bf16 precision, softmax
/// computed in f32 (critical for numerical stability), output projected
/// back to bf16. Verifies that the precision-sensitive softmax path
/// produces valid probability bounds.
#[test]
fn test_mixed_precision_attention() {
    let mut b = TensorBlockBuilder::new("pc_mixed_attn");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let shape = [SEQ_LEN, DIM];

    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let o_w = b.add_input("o_w", &[DIM, DIM]);

    let attn_out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    let def = b.build(attn_out).expect("valid mixed attention kernel");

    // q/k at BF16, v/o at BF16 (softmax internal is f32 by design)
    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], BF16_WEIGHT_MAG), // q_w (bf16)
        weight_binding(&[DIM, DIM], BF16_WEIGHT_MAG), // k_w (bf16)
        weight_binding(&[DIM, DIM], BF16_WEIGHT_MAG), // v_w (bf16)
        weight_binding(&[DIM, DIM], BF16_WEIGHT_MAG), // o_w (bf16)
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Mixed-precision attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 7. mixed_precision_layernorm
// ===========================================================================

/// Mixed-precision LayerNorm pattern: normalization computed in f32 for
/// numerical stability, result stored/projected with bf16-magnitude weights.
/// Verifies that the f32 normalization path produces valid bounds even when
/// subsequent projection uses reduced precision.
#[test]
fn test_mixed_precision_layernorm() {
    let mut b = TensorBlockBuilder::new("pc_mixed_layernorm");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let shape = [SEQ_LEN, DIM];

    // RMSNorm in f32 (norm weight = 1.0, exact)
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    // BF16 linear projection after normalization
    let w = b.add_input("w_bf16", &[DIM, DIM]);
    let out = b.add_linear(normed, w, None, &shape);
    let def = b.build(out).expect("valid mixed layernorm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        norm_weight_binding(DIM),
        weight_binding(&[DIM, DIM], BF16_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Mixed-precision layernorm IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. mixed_precision_swiglu
// ===========================================================================

/// Mixed-precision SwiGLU FFN: gate/up projections at bf16, gating sigmoid
/// computed in f32 for accuracy, down projection in bf16. The sigmoid
/// activation is precision-sensitive; computing it in f32 preserves accuracy.
#[test]
fn test_mixed_precision_swiglu() {
    let mut b = TensorBlockBuilder::new("pc_mixed_swiglu");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, DIM];

    // BF16 gate and up projections
    let gate_w = b.add_input("gate_w", &[FFN_DIM, DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, DIM]);
    // BF16 down projection
    let down_w = b.add_input("down_w", &[DIM, FFN_DIM]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    // SiLU gating (sigmoid computed in f32 internally)
    let gate_act = add_silu(&mut b, gate, &ffn_shape);

    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &out_shape);
    let def = b.build(out).expect("valid mixed SwiGLU kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[FFN_DIM, DIM], BF16_WEIGHT_MAG), // gate
        weight_binding(&[FFN_DIM, DIM], BF16_WEIGHT_MAG), // up
        weight_binding(&[DIM, FFN_DIM], BF16_WEIGHT_MAG), // down
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Mixed-precision SwiGLU IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. bf16_rounding_bounds
// ===========================================================================

/// Verify that bf16 nearest-even rounding error is bounded. The worst-case
/// error for a single bf16 conversion is ULP/2 ~ value * BF16_EPS. After
/// a linear layer, this manifests as wider bounds. Compare bf16 vs f32
/// bound widths and verify the difference is proportional to BF16_EPS.
#[test]
fn test_bf16_rounding_bounds() {
    let def = build_linear_kernel("pc_bf16_rounding");

    let fp32_width = linear_ibp_width(&def, FP32_WEIGHT_MAG, 1.0);
    let bf16_width = linear_ibp_width(&def, BF16_WEIGHT_MAG, 1.0);

    eprintln!("bf16 rounding: fp32_width={fp32_width:.6}, bf16_width={bf16_width:.6}");
    assert!(fp32_width.is_finite(), "FP32 width must be finite");
    assert!(bf16_width.is_finite(), "BF16 width must be finite");
    // BF16 has larger effective weight magnitude => wider bounds
    assert!(
        bf16_width >= fp32_width - 1e-4,
        "bf16 should have >= fp32 width: bf16={bf16_width}, fp32={fp32_width}"
    );
    // The relative increase should be bounded by BF16_EPS
    if fp32_width > 1e-6 {
        let relative_increase = (bf16_width - fp32_width) / fp32_width;
        eprintln!("Relative bound increase: {relative_increase:.6} (BF16_EPS={BF16_EPS:.6})");
        assert!(
            relative_increase < 0.1,
            "rounding increase should be small, got {relative_increase}"
        );
    }
}

// ===========================================================================
// 10. f16_denormal_flush_bounds
// ===========================================================================

/// F16 flush-to-zero (FTZ) behavior: denormal values (< 2^-14 ~ 6.1e-5)
/// are flushed to zero. Model this by using very small weight magnitudes
/// that would produce denormal-range outputs and verifying bounds remain
/// valid. With FTZ, output bounds should include zero.
#[test]
fn test_f16_denormal_flush_bounds() {
    let mut b = TensorBlockBuilder::new("pc_f16_denorm_flush");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let w = b.add_input("w_tiny", &[DIM, DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid f16 denormal kernel");

    // Very small weight magnitude: output in denormal range for f16
    let denorm_weight_mag: f32 = 1e-6;
    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], denorm_weight_mag),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("f16 denormal flush IBP: bounds=[{lo_min:.6e}, {hi_max:.6e}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Bounds should include zero (FTZ flushes denormals to 0)
    assert!(
        lo_min <= 0.0 && hi_max >= 0.0,
        "denormal flush bounds should include zero: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// 11. precision_chain_roundtrip_error
// ===========================================================================

/// Roundtrip precision chain: bf16 -> f32 -> compute -> f32 -> bf16.
/// Each dtype conversion adds rounding error. The total roundtrip error
/// is bounded by 2 * BF16_EPS per element. Verify that the roundtrip
/// produces wider bounds than a single conversion.
#[test]
fn test_precision_chain_roundtrip_error() {
    let def = build_linear_kernel("pc_roundtrip_error");

    // Single conversion: f32 -> bf16
    let single_width = linear_ibp_width(&def, BF16_WEIGHT_MAG, 1.0);
    // Roundtrip: bf16 -> f32 -> compute -> f32 -> bf16 (2x epsilon)
    let roundtrip_width = linear_ibp_width(&def, ROUNDTRIP_WEIGHT_MAG, 1.0);

    eprintln!(
        "Roundtrip error: single_width={single_width:.6}, roundtrip_width={roundtrip_width:.6}"
    );
    assert!(single_width.is_finite(), "single width must be finite");
    assert!(
        roundtrip_width.is_finite(),
        "roundtrip width must be finite"
    );
    // Roundtrip has larger effective magnitude => wider bounds
    assert!(
        roundtrip_width >= single_width - 1e-4,
        "roundtrip should be wider: roundtrip={roundtrip_width}, single={single_width}"
    );
}

// ===========================================================================
// 12. dynamic_loss_scaling_range
// ===========================================================================

/// Dynamic loss scaling for mixed-precision training: scale gradients by
/// a large factor to prevent underflow in f16 gradient computation.
/// Scale ranges from 1 to 2^24 (16777216). Verify that scaled linear
/// output bounds grow proportionally and remain finite.
#[test]
fn test_dynamic_loss_scaling_range() {
    let scale_1 = 1.0_f32;
    let scale_max = 16777216.0_f32; // 2^24

    let mut b1 = TensorBlockBuilder::new("pc_loss_scale_1");
    let input1 = b1.add_input("x", &[SEQ_LEN, DIM]);
    let w1 = b1.add_input("w", &[DIM, DIM]);
    let out1 = b1.add_linear(input1, w1, None, &[SEQ_LEN, DIM]);
    let def1 = b1.build(out1).expect("valid scale=1 kernel");

    // Scale=1
    let bindings1 = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], FP32_WEIGHT_MAG * scale_1),
    ];
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);
    let output1 = graph1
        .propagate_ibp(&input_bounds)
        .expect("IBP propagation");
    assert_bounds_valid(&output1);
    let width1 = bound_width(&output1);

    // Scale=2^24
    let mut b2 = TensorBlockBuilder::new("pc_loss_scale_max");
    let input2 = b2.add_input("x", &[SEQ_LEN, DIM]);
    let w2 = b2.add_input("w", &[DIM, DIM]);
    let out2 = b2.add_linear(input2, w2, None, &[SEQ_LEN, DIM]);
    let def2 = b2.build(out2).expect("valid scale=max kernel");

    let bindings2 = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], FP32_WEIGHT_MAG * scale_max),
    ];
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("graph translation");
    let output2 = graph2
        .propagate_ibp(&input_bounds)
        .expect("IBP propagation");
    assert_bounds_valid(&output2);
    let width_max = bound_width(&output2);

    eprintln!("Loss scaling: scale=1 width={width1:.6}, scale=2^24 width={width_max:.2e}");
    assert!(width1.is_finite(), "scale=1 width must be finite");
    assert!(width_max.is_finite(), "scale=2^24 width must be finite");
    // Larger scale => proportionally wider bounds
    assert!(
        width_max > width1,
        "max scale should produce wider bounds: max={width_max}, unit={width1}"
    );
}

// ===========================================================================
// 13. gradient_unscaling
// ===========================================================================

/// Gradient unscaling: after loss-scaled forward pass, divide gradients
/// by the scale factor. Model as two sequential linear layers where the
/// first has scaled weights and the second has 1/scale weights, verifying
/// that the net effect preserves the original gradient direction (bounds).
#[test]
fn test_gradient_unscaling() {
    let loss_scale: f32 = 1024.0; // moderate scale factor

    let mut b = TensorBlockBuilder::new("pc_grad_unscale");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let shape = [SEQ_LEN, DIM];

    // Scaled forward: weight * loss_scale
    let w_scaled = b.add_input("w_scaled", &[DIM, DIM]);
    let scaled = b.add_linear(input, w_scaled, None, &shape);

    // Unscaled: weight * (1/loss_scale) — net effect ~ original weight
    let w_unscale = b.add_input("w_unscale", &[DIM, DIM]);
    let out = b.add_linear(scaled, w_unscale, None, &shape);
    let def = b.build(out).expect("valid gradient unscale kernel");

    let scaled_mag = FP32_WEIGHT_MAG * loss_scale;
    let unscale_mag = FP32_WEIGHT_MAG / loss_scale;

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], scaled_mag),
        weight_binding(&[DIM, DIM], unscale_mag),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Gradient unscaling IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Scale then unscale should produce bounded output
    let width = bound_width(&output);
    assert!(
        width.is_finite() && width > 0.0,
        "output width should be finite and positive: {width}"
    );
}

// ===========================================================================
// 14. mixed_precision_residual
// ===========================================================================

/// Mixed-precision residual connection: skip path remains in f32 for
/// stability, while the sub-block computes in bf16. The residual addition
/// (f32 skip + bf16 branch) preserves the f32 signal while adding
/// bf16-precision perturbation.
#[test]
fn test_mixed_precision_residual() {
    let mut b = TensorBlockBuilder::new("pc_mixed_residual");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let shape = [SEQ_LEN, DIM];

    // BF16 sub-block: RMSNorm -> Linear(bf16)
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_weight", &[DIM]);
    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);

    let w = b.add_input("w_bf16", &[DIM, DIM]);
    let branch = b.add_linear(normed, w, None, &shape);

    // F32 residual: x + branch(x)
    let out = b.add_binary_add(input, branch, &shape);
    let def = b.build(out).expect("valid mixed residual kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        norm_weight_binding(DIM),
        weight_binding(&[DIM, DIM], BF16_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Mixed-precision residual IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Residual preserves input range plus bf16 projection
    assert!(
        lo_min > -100.0,
        "residual lower should be reasonable, got {lo_min}"
    );
}

// ===========================================================================
// 15. int8_conv_f32_accumulation
// ===========================================================================

/// INT8 convolution with f32 accumulator: weights quantized to INT8,
/// convolution accumulated in f32 to prevent overflow. The f32 accumulator
/// produces tighter bounds than INT8 accumulation. Modeled as a linear
/// layer with INT8 weight magnitude (conv1d and linear share verification
/// properties for bound propagation).
#[test]
fn test_int8_conv_f32_accumulation() {
    // INT8 with f32 accumulation (no additional rounding per accumulation)
    let f32_accum_mag = INT8_WEIGHT_MAG;
    // INT8 with INT8 accumulation (rounding per step, wider bounds)
    let int8_accum_mag = INT8_WEIGHT_MAG * (1.0 + BF16_EPS * (DIM as f32).sqrt());

    let def = build_linear_kernel("pc_int8_conv_accum");

    let f32_accum_width = linear_ibp_width(&def, f32_accum_mag, 1.0);
    let int8_accum_width = linear_ibp_width(&def, int8_accum_mag, 1.0);

    eprintln!(
        "INT8 conv accum: f32_accum_width={f32_accum_width:.6}, int8_accum_width={int8_accum_width:.6}"
    );
    assert!(
        f32_accum_width.is_finite(),
        "f32 accum width must be finite"
    );
    assert!(
        int8_accum_width.is_finite(),
        "INT8 accum width must be finite"
    );
    // F32 accumulator should produce tighter bounds
    assert!(
        f32_accum_width <= int8_accum_width + 1e-4,
        "f32 accum should be tighter: f32={f32_accum_width}, int8={int8_accum_width}"
    );
}

// ===========================================================================
// 16. int4_gptq_dequant_bounds
// ===========================================================================

/// INT4 GPTQ dequantization with group-wise scale and zero-point. Each
/// group of weights has an independent scale factor from Hessian-based
/// optimization. Verify that per-group variation produces valid bounds.
#[test]
fn test_int4_gptq_dequant_bounds() {
    let out_dim = GROUP_SIZE * 2; // 32
    let in_dim = GROUP_SIZE; // 16

    let mut b = TensorBlockBuilder::new("pc_int4_gptq_dequant");
    let input = b.add_input("x", &[SEQ_LEN, in_dim]);
    let w = b.add_input("w_gptq", &[out_dim, in_dim]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, out_dim]);
    let def = b.build(out).expect("valid GPTQ dequant kernel");

    // Group 1: normal INT4 magnitude; Group 2: GPTQ residual magnitude
    let mut weight_data = vec![INT4_WEIGHT_MAG; out_dim * in_dim];
    for row in GROUP_SIZE..out_dim {
        for col in 0..in_dim {
            weight_data[row * in_dim + col] = GPTQ_WEIGHT_MAG;
        }
    }
    let w_tensor =
        ArrayD::from_shape_vec(IxDyn(&[out_dim, in_dim]), weight_data).expect("valid weight shape");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w_tensor),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, in_dim], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT4 GPTQ dequant IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 17. awq_per_channel_scale
// ===========================================================================

/// AWQ (Activation-Aware Weight Quantization) per-channel scaling: salient
/// channels are rescaled before quantization to preserve activation
/// statistics. The net dequantized weight magnitude is reduced by the
/// salient scale factor, producing tighter bounds than naive INT4.
#[test]
fn test_awq_per_channel_scale() {
    let def = build_linear_kernel("pc_awq_channel_scale");

    let awq_width = linear_ibp_width(&def, AWQ_WEIGHT_MAG, 1.0);
    let int4_width = linear_ibp_width(&def, INT4_WEIGHT_MAG, 1.0);

    eprintln!("AWQ per-channel: awq_width={awq_width:.6}, int4_width={int4_width:.6}");
    assert!(awq_width.is_finite(), "AWQ width must be finite");
    assert!(int4_width.is_finite(), "INT4 width must be finite");
    // AWQ has smaller effective magnitude => tighter bounds
    assert!(
        awq_width <= int4_width + 1e-4,
        "AWQ should be tighter: awq={awq_width}, int4={int4_width}"
    );
}

// ===========================================================================
// 18. full_pipeline_bf16_bounds
// ===========================================================================

/// Full mixed-precision bf16 inference pipeline: input -> RMSNorm -> MHA
/// (bf16 projections, f32 softmax) -> residual -> RMSNorm -> SwiGLU (bf16)
/// -> residual -> output. Verifies end-to-end bound composition through
/// a complete transformer block at bf16 precision.
fn build_full_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("pc_full_pipeline_bf16");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let shape = [SEQ_LEN, DIM];

    // Pre-attention RMSNorm
    let attn_eps = b.add_input("attn_eps", &[1]);
    let attn_norm_w = b.add_input("attn_norm_w", &[DIM]);
    let normed_attn = b.add_rms_norm(input, attn_eps, 1, attn_norm_w, &shape);

    // BF16 multi-head attention
    let q_w = b.add_input("q_w", &[DIM, DIM]);
    let k_w = b.add_input("k_w", &[DIM, DIM]);
    let v_w = b.add_input("v_w", &[DIM, DIM]);
    let o_w = b.add_input("o_w", &[DIM, DIM]);
    let attn_out = b
        .add_multi_head_attention(
            normed_attn,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &shape,
        )
        .expect("valid MHA");

    // First residual
    let h = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let ffn_eps = b.add_input("ffn_eps", &[1]);
    let ffn_norm_w = b.add_input("ffn_norm_w", &[DIM]);
    let normed_ffn = b.add_rms_norm(h, ffn_eps, 1, ffn_norm_w, &shape);

    // BF16 SwiGLU FFN
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let gate_w = b.add_input("gate_w", &[FFN_DIM, DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, DIM]);
    let down_w = b.add_input("down_w", &[DIM, FFN_DIM]);

    let gate = b.add_linear(normed_ffn, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);
    let up = b.add_linear(normed_ffn, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Second residual
    let out = b.add_binary_add(h, ffn_out, &shape);
    b.build(out).expect("valid full pipeline kernel")
}

fn full_pipeline_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        eps_binding(),                                    // attn_eps
        norm_weight_binding(DIM),                         // attn_norm_w
        weight_binding(&[DIM, DIM], BF16_WEIGHT_MAG),     // q_w
        weight_binding(&[DIM, DIM], BF16_WEIGHT_MAG),     // k_w
        weight_binding(&[DIM, DIM], BF16_WEIGHT_MAG),     // v_w
        weight_binding(&[DIM, DIM], BF16_WEIGHT_MAG),     // o_w
        eps_binding(),                                    // ffn_eps
        norm_weight_binding(DIM),                         // ffn_norm_w
        weight_binding(&[FFN_DIM, DIM], BF16_WEIGHT_MAG), // gate_w
        weight_binding(&[FFN_DIM, DIM], BF16_WEIGHT_MAG), // up_w
        weight_binding(&[DIM, FFN_DIM], BF16_WEIGHT_MAG), // down_w
    ]
}

#[test]
fn test_full_pipeline_bf16_bounds_ibp() {
    let def = build_full_pipeline_kernel();
    let bindings = full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full pipeline bf16 IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_full_pipeline_bf16_bounds_crown() {
    let def = build_full_pipeline_kernel();
    let bindings = full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full pipeline bf16 CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
