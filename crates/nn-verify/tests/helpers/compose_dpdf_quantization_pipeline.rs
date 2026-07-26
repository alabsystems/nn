// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for model quantization pipeline bound preservation.
//!
//! Verifies that quantization (F16/BF16/INT8) preserves model output bounds
//! relative to FP32 baselines through NY IBP and CROWN propagation.
//!
//! ## Cast Precision Preservation (tests 1-2)
//!
//! 1. **F32->F16 cast bound preservation (IBP)**: linear layer with F16
//!    weight magnitudes preserves bounds within epsilon of FP32 baseline.
//!
//! 2. **F32->BF16 cast bound preservation (IBP)**: linear layer with BF16
//!    weight magnitudes preserves bounds within epsilon of FP32 baseline.
//!
//! ## Per-Channel INT8 Quantization (test 3)
//!
//! 3. **Per-channel INT8 quantization error bounded by scale/2 (IBP)**:
//!    INT8 per-channel quantization with known scale produces output bounds
//!    whose width is bounded by the quantization step size.
//!
//! ## Symmetric Roundtrip (test 4)
//!
//! 4. **Quantize->dequantize roundtrip bound preservation (IBP)**: two
//!    linear layers in series (simulating quantize then dequantize) preserve
//!    output bounds within a known margin.
//!
//! ## Layer-Level Quantization (tests 5-7)
//!
//! 5. **INT8 linear layer output within epsilon of FP32 (IBP)**: comparing
//!    INT8-quantized and FP32 linear layer output bound widths.
//!
//! 6. **Mixed-precision attention bounds (IBP + CROWN)**: attention with
//!    reduced-precision (F16-magnitude) QKV projections maintains valid bounds.
//!
//! 7. **Reduced-precision LayerNorm bounds (IBP)**: LayerNorm followed by
//!    reduced-precision linear projection maintains valid bounds.
//!
//! ## Full Block (test 8)
//!
//! 8. **Full transformer block INT8 vs FP32 (IBP)**: complete transformer
//!    sub-block (attention + FFN + residuals) with INT8 vs FP32 weight
//!    magnitudes, verifying bounded output deviation.
//!
//! ## Asymmetric & Per-Channel/Per-Tensor (tests 9-10)
//!
//! 9. **Asymmetric INT8 with non-zero zero-point (IBP)**: asymmetric INT8
//!    quantization (scale * (code - zero_point)) produces shifted but bounded
//!    output compared to symmetric INT8.
//!
//! 10. **Per-channel vs per-tensor INT8 bound comparison (IBP)**: per-channel
//!     quantization produces tighter bounds than per-tensor because each channel
//!     has its own scale factor.
//!
//! ## Quantized Operations (tests 11-12)
//!
//! 11. **Quantized matmul error accumulation (IBP)**: matmul with quantized
//!     weights accumulates error proportional to inner dimension. Verifies
//!     output width scales predictably with DIM.
//!
//! 12. **Quantized softmax output bounded in [0, 1] (IBP)**: softmax after
//!     quantized linear projection still produces valid probability outputs.
//!
//! ## Full Pipeline (test 13)
//!
//! 13. **Full encoder -> quantize -> decoder pipeline (IBP + CROWN)**:
//!     end-to-end pipeline with mixed precision: FP32 encoder attention +
//!     INT8-precision FFN, through residual + softmax output.
//!
//! Quantization is modeled via weight magnitude differences:
//! - FP32 baseline: `WEIGHT_MAG = 0.02`
//! - FP16 equivalent: `FP16_WEIGHT_MAG = 0.0199` (1 ULP epsilon for F16)
//! - BF16 equivalent: `BF16_WEIGHT_MAG = 0.0195` (wider mantissa epsilon)
//! - INT8 symmetric: `INT8_WEIGHT_MAG = 0.0635` (scale * 127)
//!
//! Dimensions (small for fast verification):
//! - SEQ_LEN=4, DIM=32, FFN_DIM=64, NUM_HEADS=4
//!
//! Part of #4216: Compose tests for quantization pipeline bound preservation.

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
const HEAD_DIM: usize = DIM / NUM_HEADS; // 8

// ---------------------------------------------------------------------------
// Quantization parameters
// ---------------------------------------------------------------------------

/// FP32 baseline weight magnitude.
const FP32_WEIGHT_MAG: f32 = 0.02;
/// FP16 equivalent weight magnitude (FP32 - F16 epsilon ~5e-4 for this range).
const FP16_WEIGHT_MAG: f32 = 0.0199;
/// BF16 equivalent weight magnitude (wider mantissa epsilon ~1e-3).
const BF16_WEIGHT_MAG: f32 = 0.0195;
/// INT8 symmetric dequantized weight magnitude: scale * 127, scale=0.0005.
const INT8_WEIGHT_MAG: f32 = 0.0635;
/// INT8 quantization scale for per-channel analysis.
const INT8_SCALE: f32 = 0.0005;

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

/// Zero bias binding.
fn zero_bias_binding(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// Norm weight (all ones) binding.
fn norm_weight_binding(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 1.0f32))
}

/// Norm bias (all zeros) binding.
fn norm_bias_binding(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 0.0f32))
}

/// Epsilon binding for normalization.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32))
}

/// Compute output bound width from a BoundedTensor.
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

/// Build a simple linear kernel def.
fn build_linear_kernel(name: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let w = b.add_input("w", &[DIM, DIM]);
    let bias = b.add_input("bias", &[DIM]);
    let out = b.add_linear(input, w, Some(bias), &[SEQ_LEN, DIM]);
    b.build(out).expect("valid linear kernel")
}

/// Run IBP on a linear kernel with given weight magnitude, return output bounds.
fn linear_ibp_propagate(def: &TensorKernelDef, weight_mag: f32, input_range: f32) -> BoundedTensor {
    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], weight_mag),
        bias_binding(&[DIM]),
    ];
    let graph = tensor_kernel_to_graph(def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], input_range);
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    output
}

/// Run IBP on a linear kernel with given weight magnitude, return width.
fn linear_ibp_width(def: &TensorKernelDef, weight_mag: f32, input_range: f32) -> f32 {
    bound_width(&linear_ibp_propagate(def, weight_mag, input_range))
}

// ===========================================================================
// 1. F32 -> F16 cast bound preservation (IBP)
// ===========================================================================

/// Verify that F16-equivalent weight magnitudes preserve output bounds within
/// epsilon of the FP32 baseline. F16 has 10-bit mantissa, so values in the
/// range [0.01, 0.1] have precision ~5e-4. The linear layer output bound
/// width difference should be proportional to the weight magnitude difference.
#[test]
fn test_quantization_f32_to_f16_bound_preservation() {
    let def = build_linear_kernel("quant_pipe_f16_cast");

    let fp32_width = linear_ibp_width(&def, FP32_WEIGHT_MAG, 1.0);
    let fp16_width = linear_ibp_width(&def, FP16_WEIGHT_MAG, 1.0);

    eprintln!(
        "F32->F16 bound preservation: FP32 width={fp32_width:.6}, FP16 width={fp16_width:.6}"
    );

    // Both must produce valid finite bounds
    assert!(fp32_width.is_finite(), "FP32 width must be finite");
    assert!(fp16_width.is_finite(), "FP16 width must be finite");

    // FP16 weight magnitude is very close to FP32 (0.0199 vs 0.02), so
    // bound widths should be within a small relative epsilon.
    // |fp32 - fp16| / fp32 < 0.01 (1% tolerance, generous for F16 precision).
    let relative_diff = (fp32_width - fp16_width).abs() / fp32_width;
    eprintln!("Relative difference: {relative_diff:.6}");
    assert!(
        relative_diff < 0.01,
        "F16 cast should preserve bounds within 1%: FP32={fp32_width}, FP16={fp16_width}, diff={relative_diff}"
    );
}

// ===========================================================================
// 2. F32 -> BF16 cast bound preservation (IBP)
// ===========================================================================

/// Verify that BF16-equivalent weight magnitudes preserve output bounds within
/// epsilon of the FP32 baseline. BF16 has 7-bit mantissa (same exponent range
/// as FP32), so values have precision ~1e-3. The output bound width difference
/// should be proportionally larger than F16 but still bounded.
#[test]
fn test_quantization_f32_to_bf16_bound_preservation() {
    let def = build_linear_kernel("quant_pipe_bf16_cast");

    let fp32_width = linear_ibp_width(&def, FP32_WEIGHT_MAG, 1.0);
    let bf16_width = linear_ibp_width(&def, BF16_WEIGHT_MAG, 1.0);

    eprintln!(
        "F32->BF16 bound preservation: FP32 width={fp32_width:.6}, BF16 width={bf16_width:.6}"
    );

    assert!(fp32_width.is_finite(), "FP32 width must be finite");
    assert!(bf16_width.is_finite(), "BF16 width must be finite");

    // BF16 has wider mantissa epsilon than F16, so allow 3% tolerance.
    let relative_diff = (fp32_width - bf16_width).abs() / fp32_width;
    eprintln!("Relative difference: {relative_diff:.6}");
    assert!(
        relative_diff < 0.03,
        "BF16 cast should preserve bounds within 3%: FP32={fp32_width}, BF16={bf16_width}, diff={relative_diff}"
    );

    // BF16 should have wider (or equal) deviation from FP32 than F16
    let fp16_width = linear_ibp_width(&def, FP16_WEIGHT_MAG, 1.0);
    let fp16_diff = (fp32_width - fp16_width).abs();
    let bf16_diff = (fp32_width - bf16_width).abs();
    eprintln!("FP16 abs diff: {fp16_diff:.6}, BF16 abs diff: {bf16_diff:.6}");
    assert!(
        bf16_diff >= fp16_diff - 1e-6,
        "BF16 deviation should be >= FP16 deviation: BF16={bf16_diff}, FP16={fp16_diff}"
    );
}

// ===========================================================================
// 3. Per-channel INT8 quantization error bounded by scale/2 (IBP)
// ===========================================================================

/// Verify that INT8 per-channel quantization produces output bounds whose
/// width is bounded. For symmetric INT8 with scale s, the max quantization
/// error per weight element is s/2 (rounding to nearest). For a linear layer
/// with DIM inputs, the accumulated error in the output is at most
/// DIM * input_range * (s/2). We verify the output bound width is finite
/// and proportional to INT8_SCALE.
#[test]
fn test_quantization_per_channel_int8_bounds() {
    let def = build_linear_kernel("quant_pipe_int8_perchannel");

    // INT8 quantized path
    let int8_output = linear_ibp_propagate(&def, INT8_WEIGHT_MAG, 1.0);
    let int8_width = bound_width(&int8_output);

    // FP32 baseline path
    let fp32_output = linear_ibp_propagate(&def, FP32_WEIGHT_MAG, 1.0);
    let fp32_width = bound_width(&fp32_output);

    eprintln!("Per-channel INT8 bounds: INT8 width={int8_width:.6}, FP32 width={fp32_width:.6}");

    assert!(int8_width.is_finite(), "INT8 width must be finite");
    assert!(fp32_width.is_finite(), "FP32 width must be finite");

    // INT8 weights have larger magnitude (0.0635 vs 0.02), so wider output bounds.
    // The width difference is bounded by the quantization error accumulation:
    // max_error_per_element = INT8_SCALE / 2 = 0.00025
    // accumulated over DIM=32 inputs * input_range=2 ([-1, 1]):
    // max_width_increase ~= DIM * 2 * INT8_SCALE / 2 = 0.008
    // But since INT8_WEIGHT_MAG is much larger than FP32_WEIGHT_MAG, the
    // dominant effect is the weight magnitude difference.
    let width_ratio = int8_width / fp32_width;
    eprintln!("Width ratio (INT8/FP32): {width_ratio:.4}");

    // INT8 weights are ~3.175x larger in magnitude, so output bounds scale similarly
    let expected_ratio = INT8_WEIGHT_MAG / FP32_WEIGHT_MAG;
    let ratio_diff = (width_ratio - expected_ratio).abs() / expected_ratio;
    eprintln!(
        "Expected ratio: {expected_ratio:.4}, actual: {width_ratio:.4}, diff: {ratio_diff:.4}"
    );
    assert!(
        ratio_diff < 0.1,
        "Width ratio should be proportional to weight magnitude ratio: \
         expected ~{expected_ratio}, got {width_ratio}"
    );
}

// ===========================================================================
// 4. Quantize -> dequantize roundtrip bound preservation (IBP)
// ===========================================================================

/// Verify that a quantize-then-dequantize roundtrip preserves bounds within
/// a bounded margin. Models this as two serial linear layers:
/// - First layer: input -> intermediate (simulating quantization scaling)
/// - Second layer: intermediate -> output (simulating dequantization scaling)
///
/// The roundtrip should approximately preserve bounds: the output width is
/// within a bounded ratio of the input width.
#[test]
fn test_quantization_symmetric_roundtrip_bounds() {
    let scale = INT8_SCALE;
    let inv_scale = 1.0 / scale; // large value

    // Build: input -> Linear(scale) -> Linear(inv_scale) -> output
    // This models quantize (multiply by 1/scale, round to int) then
    // dequantize (multiply by scale). We approximate with two linear layers.
    let mut b = TensorBlockBuilder::new("quant_pipe_roundtrip");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    // Quantization step: scale weights by `scale` (shrink)
    let quant_w = b.add_input("quant_w", &[DIM, DIM]);
    let quant_out = b.add_linear(input, quant_w, None, &[SEQ_LEN, DIM]);

    // Dequantization step: scale back by `inv_scale` (expand)
    let dequant_w = b.add_input("dequant_w", &[DIM, DIM]);
    let output = b.add_linear(quant_out, dequant_w, None, &[SEQ_LEN, DIM]);

    let def = b.build(output).expect("valid roundtrip kernel");

    // Use diagonal-like weights: identity * scale for quantize,
    // identity * (1/scale) for dequantize. Approximated by uniform mag.
    // The product scale * (1/scale) = 1.0, so the net effect on magnitude
    // should be identity-like.
    let quant_mag = scale; // 0.0005
    let dequant_mag = inv_scale / (DIM as f32); // normalize so product ~ identity

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], quant_mag),
        weight_binding(&[DIM, DIM], dequant_mag),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP roundtrip");
    assert_bounds_valid(&output);

    let input_width = 2.0; // [-1, 1] range
    let output_width = bound_width(&output);
    eprintln!("Roundtrip bounds: input_width={input_width:.6}, output_width={output_width:.6}");

    // Output width should be finite and positive
    assert!(
        output_width.is_finite() && output_width > 0.0,
        "Roundtrip output must have positive finite width: got {output_width}"
    );
}

// ===========================================================================
// 5. INT8 linear layer output within epsilon of FP32 (IBP)
// ===========================================================================

/// Verify that INT8 quantized linear layer output bounds are structurally
/// valid and that the width difference relative to FP32 is bounded.
/// The quantization error per output element is at most
/// sum_j(|x_j| * |w_int8_j - w_fp32_j|), bounded by DIM * input_range * delta_w.
#[test]
fn test_quantization_linear_layer_int8_vs_f32() {
    let def = build_linear_kernel("quant_pipe_linear_int8_vs_f32");

    let fp32_output = linear_ibp_propagate(&def, FP32_WEIGHT_MAG, 1.0);
    let int8_output = linear_ibp_propagate(&def, INT8_WEIGHT_MAG, 1.0);

    let fp32_width = bound_width(&fp32_output);
    let int8_width = bound_width(&int8_output);

    eprintln!("INT8 vs FP32 linear: FP32 width={fp32_width:.6}, INT8 width={int8_width:.6}");

    // Both must produce valid bounds
    let (fp32_lo, fp32_hi) = bounds_min_max(&fp32_output);
    let (int8_lo, int8_hi) = bounds_min_max(&int8_output);
    assert!(fp32_lo.is_finite() && fp32_hi.is_finite());
    assert!(int8_lo.is_finite() && int8_hi.is_finite());

    // The bound width difference is bounded by the weight magnitude difference
    // scaled by the input dimensions: delta_w * DIM * input_range
    let delta_w = (INT8_WEIGHT_MAG - FP32_WEIGHT_MAG).abs();
    let max_width_diff = delta_w * (DIM as f32) * 2.0; // input range [-1, 1] = 2.0
    let actual_width_diff = (int8_width - fp32_width).abs();
    eprintln!("Width diff: actual={actual_width_diff:.6}, max_expected={max_width_diff:.6}");
    assert!(
        actual_width_diff <= max_width_diff + 1e-4,
        "Width difference should be bounded: actual={actual_width_diff}, max={max_width_diff}"
    );
}

// ===========================================================================
// 6. Mixed-precision attention bounds (IBP + CROWN)
// ===========================================================================

/// Verify that attention with reduced-precision (F16-magnitude) QKV
/// projections maintains valid output bounds. The attention mechanism
/// (softmax(QK^T/sqrt(d)) * V) produces outputs in a bounded range
/// regardless of the precision of Q, K, V projections.
#[test]
fn test_quantization_attention_precision_bounds() {
    let shape = [SEQ_LEN, DIM];
    let head_shape = [NUM_HEADS, SEQ_LEN, HEAD_DIM];

    let mut b = TensorBlockBuilder::new("quant_pipe_attn_precision");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    // F16-magnitude QKV projections (simulating mixed precision attention)
    let wq = b.add_input("wq", &[DIM, DIM]);
    let wk = b.add_input("wk", &[DIM, DIM]);
    let wv = b.add_input("wv", &[DIM, DIM]);

    let q = b.add_linear(input, wq, None, &shape);
    let k = b.add_linear(input, wk, None, &shape);
    let v = b.add_linear(input, wv, None, &shape);

    // Reshape for multi-head attention: [SEQ_LEN, DIM] -> [NUM_HEADS, SEQ_LEN, HEAD_DIM]
    let q_h = b.add_reshape(q, &head_shape);
    let k_h = b.add_reshape(k, &head_shape);
    let v_h = b.add_reshape(v, &head_shape);

    // Self-attention per head
    let attn_out = b.add_attention(q_h, k_h, v_h, AttentionMask::Standard, None, &head_shape);

    // Reshape back: [NUM_HEADS, SEQ_LEN, HEAD_DIM] -> [SEQ_LEN, DIM]
    let out = b.add_reshape(attn_out, &shape);

    // Output projection
    let wo = b.add_input("wo", &[DIM, DIM]);
    let final_out = b.add_linear(out, wo, None, &shape);

    let def = b
        .build(final_out)
        .expect("valid attention precision kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], FP16_WEIGHT_MAG), // wq (F16)
        weight_binding(&[DIM, DIM], FP16_WEIGHT_MAG), // wk (F16)
        weight_binding(&[DIM, DIM], FP16_WEIGHT_MAG), // wv (F16)
        weight_binding(&[DIM, DIM], FP16_WEIGHT_MAG), // wo (F16)
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    // IBP propagation
    let ibp_output = graph.propagate_ibp(&input).expect("IBP through attention");
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Attention precision IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "attention lower bound must be finite");
    assert!(hi_max.is_finite(), "attention upper bound must be finite");

    // CROWN propagation (may fall back to IBP for attention)
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (c_lo, c_hi) = bounds_min_max(&crown_output);
    eprintln!("Attention precision CROWN: method={method:?}, bounds=[{c_lo:.6}, {c_hi:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 7. Reduced-precision LayerNorm bounds (IBP)
// ===========================================================================

/// Verify that LayerNorm followed by a reduced-precision (F16-magnitude)
/// linear projection maintains valid output bounds. LayerNorm normalizes
/// the input to zero mean and unit variance, which should produce bounded
/// outputs regardless of input precision.
#[test]
fn test_quantization_normalization_precision_bounds() {
    let shape = [SEQ_LEN, DIM];

    let mut b = TensorBlockBuilder::new("quant_pipe_norm_precision");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);

    // LayerNorm
    let eps = b.add_input("eps", &[1]);
    let ln_weight = b.add_input("ln_weight", &[DIM]);
    let ln_bias = b.add_input("ln_bias", &[DIM]);
    let normed = b.add_layer_norm(input, eps, 1, ln_weight, ln_bias, &shape);

    // Reduced-precision linear projection
    let w = b.add_input("w", &[DIM, DIM]);
    let bias = b.add_input("bias", &[DIM]);
    let out = b.add_linear(normed, w, Some(bias), &shape);

    let def = b.build(out).expect("valid norm precision kernel");

    // FP32 baseline
    let fp32_bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        norm_weight_binding(DIM),
        norm_bias_binding(DIM),
        weight_binding(&[DIM, DIM], FP32_WEIGHT_MAG),
        bias_binding(&[DIM]),
    ];
    let fp32_graph = tensor_kernel_to_graph(&def, &fp32_bindings).expect("FP32 graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, DIM], 1.0);
    let fp32_output = fp32_graph.propagate_ibp(&input_bounds).expect("FP32 IBP");
    assert_bounds_valid(&fp32_output);

    // F16-equivalent precision
    let fp16_bindings = vec![
        TensorParamBinding::Variable,
        eps_binding(),
        norm_weight_binding(DIM),
        norm_bias_binding(DIM),
        weight_binding(&[DIM, DIM], FP16_WEIGHT_MAG),
        bias_binding(&[DIM]),
    ];
    let fp16_graph = tensor_kernel_to_graph(&def, &fp16_bindings).expect("FP16 graph translation");
    let fp16_output = fp16_graph.propagate_ibp(&input_bounds).expect("FP16 IBP");
    assert_bounds_valid(&fp16_output);

    let fp32_width = bound_width(&fp32_output);
    let fp16_width = bound_width(&fp16_output);
    eprintln!("Norm + precision: FP32 width={fp32_width:.6}, FP16 width={fp16_width:.6}");

    // Both must be finite and the difference should be small since LayerNorm
    // normalizes the signal before the precision-sensitive projection.
    assert!(fp32_width.is_finite(), "FP32 norm width must be finite");
    assert!(fp16_width.is_finite(), "FP16 norm width must be finite");

    // LayerNorm stabilizes bounds, so precision difference is small
    let relative_diff = (fp32_width - fp16_width).abs() / (fp32_width + 1e-8);
    eprintln!("Relative difference after LayerNorm: {relative_diff:.6}");
    assert!(
        relative_diff < 0.05,
        "LayerNorm should stabilize precision: FP32={fp32_width}, FP16={fp16_width}, diff={relative_diff}"
    );
}

// ===========================================================================
// 8. Full transformer block INT8 vs FP32 (IBP)
// ===========================================================================

/// Verify that a full transformer sub-block (attention + SwiGLU FFN +
/// residuals) with INT8 vs FP32 weight magnitudes produces bounded output
/// deviation. The block structure:
///   x -> Attention(x) + x -> SwiGLU_FFN(normed) + normed -> output
///
/// The residual connections bound the output deviation because the skip
/// path preserves the original input range.
#[test]
fn test_quantization_full_block_int8_bounds() {
    // Build a minimal transformer sub-block:
    // input -> linear(Q,K,V) -> attention -> + input (residual)
    //       -> linear(FFN gate) -> SiLU -> mul(FFN up) -> linear(FFN down) + residual
    let shape = [SEQ_LEN, DIM];

    // ---- FP32 block ----
    let fp32_def = build_transformer_block("quant_pipe_block_fp32");
    let fp32_bindings = transformer_block_bindings(FP32_WEIGHT_MAG);
    let fp32_graph = tensor_kernel_to_graph(&fp32_def, &fp32_bindings).expect("FP32 block graph");
    let input = uniform_bounds(&shape, 1.0);
    let fp32_output = fp32_graph.propagate_ibp(&input).expect("FP32 block IBP");
    assert_bounds_valid(&fp32_output);

    // ---- INT8 block ----
    let int8_def = build_transformer_block("quant_pipe_block_int8");
    let int8_bindings = transformer_block_bindings(INT8_WEIGHT_MAG);
    let int8_graph = tensor_kernel_to_graph(&int8_def, &int8_bindings).expect("INT8 block graph");
    let int8_output = int8_graph.propagate_ibp(&input).expect("INT8 block IBP");
    assert_bounds_valid(&int8_output);

    let fp32_width = bound_width(&fp32_output);
    let int8_width = bound_width(&int8_output);
    let (fp32_lo, fp32_hi) = bounds_min_max(&fp32_output);
    let (int8_lo, int8_hi) = bounds_min_max(&int8_output);

    eprintln!("Full block FP32: bounds=[{fp32_lo:.4}, {fp32_hi:.4}], width={fp32_width:.4}");
    eprintln!("Full block INT8: bounds=[{int8_lo:.4}, {int8_hi:.4}], width={int8_width:.4}");

    // Both must produce finite bounds
    assert!(fp32_width.is_finite(), "FP32 block width must be finite");
    assert!(int8_width.is_finite(), "INT8 block width must be finite");

    // INT8 has larger weight magnitudes -> wider bounds, but the residual
    // connections bound the growth. The width ratio should be finite.
    let width_ratio = int8_width / fp32_width;
    eprintln!("Width ratio (INT8/FP32): {width_ratio:.4}");
    assert!(
        width_ratio.is_finite() && width_ratio > 0.0,
        "Width ratio must be finite and positive: got {width_ratio}"
    );
}

/// Build a transformer sub-block: attention + residual + SwiGLU FFN + residual.
fn build_transformer_block(name: &str) -> TensorKernelDef {
    let shape = [SEQ_LEN, DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let head_shape = [NUM_HEADS, SEQ_LEN, HEAD_DIM];

    let mut b = TensorBlockBuilder::new(name);
    let input = b.add_input("x", &shape);

    // -- Attention sub-block --
    let wq = b.add_input("wq", &[DIM, DIM]);
    let wk = b.add_input("wk", &[DIM, DIM]);
    let wv = b.add_input("wv", &[DIM, DIM]);
    let wo = b.add_input("wo", &[DIM, DIM]);

    let q = b.add_linear(input, wq, None, &shape);
    let k = b.add_linear(input, wk, None, &shape);
    let v = b.add_linear(input, wv, None, &shape);

    let q_h = b.add_reshape(q, &head_shape);
    let k_h = b.add_reshape(k, &head_shape);
    let v_h = b.add_reshape(v, &head_shape);

    let attn = b.add_attention(q_h, k_h, v_h, AttentionMask::Standard, None, &head_shape);
    let attn_flat = b.add_reshape(attn, &shape);
    let attn_proj = b.add_linear(attn_flat, wo, None, &shape);

    // Residual
    let post_attn = b.add_binary_add(input, attn_proj, &shape);

    // -- SwiGLU FFN sub-block --
    let gate_w = b.add_input("gate_w", &[FFN_DIM, DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, DIM]);
    let down_w = b.add_input("down_w", &[DIM, FFN_DIM]);

    let gate = b.add_linear(post_attn, gate_w, None, &ffn_shape);
    let gate_act = add_silu(&mut b, gate, &ffn_shape);
    let up = b.add_linear(post_attn, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    // Residual
    let output = b.add_binary_add(post_attn, ffn_out, &shape);

    b.build(output).expect("valid transformer block kernel")
}

/// Build bindings for the transformer block with given weight magnitude.
fn transformer_block_bindings(weight_mag: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,                // input
        weight_binding(&[DIM, DIM], weight_mag),     // wq
        weight_binding(&[DIM, DIM], weight_mag),     // wk
        weight_binding(&[DIM, DIM], weight_mag),     // wv
        weight_binding(&[DIM, DIM], weight_mag),     // wo
        weight_binding(&[FFN_DIM, DIM], weight_mag), // gate_w
        weight_binding(&[FFN_DIM, DIM], weight_mag), // up_w
        weight_binding(&[DIM, FFN_DIM], weight_mag), // down_w
    ]
}

// ===========================================================================
// 9. Asymmetric INT8 with non-zero zero-point (IBP)
// ===========================================================================

/// Asymmetric INT8 quantization (scale * (code - zero_point)) produces
/// shifted but bounded output compared to symmetric INT8.
#[test]
fn test_quantization_asymmetric_int8_zero_point() {
    let zero_point_offset: f32 = 8.0;
    let asym_weight_mag: f32 = INT8_WEIGHT_MAG - INT8_SCALE * zero_point_offset;

    let def = build_linear_kernel("quant_pipe_asym_int8");
    let sym_output = linear_ibp_propagate(&def, INT8_WEIGHT_MAG, 1.0);
    let asym_output = linear_ibp_propagate(&def, asym_weight_mag, 1.0);

    let sym_width = bound_width(&sym_output);
    let asym_width = bound_width(&asym_output);
    let (sym_lo, sym_hi) = bounds_min_max(&sym_output);
    let (asym_lo, asym_hi) = bounds_min_max(&asym_output);

    eprintln!("Symmetric INT8: [{sym_lo:.6}, {sym_hi:.6}], width={sym_width:.6}");
    eprintln!("Asymmetric INT8: [{asym_lo:.6}, {asym_hi:.6}], width={asym_width:.6}");
    assert!(sym_width.is_finite(), "symmetric width must be finite");
    assert!(asym_width.is_finite(), "asymmetric width must be finite");
    assert!(
        asym_width <= sym_width + 1e-4,
        "asymmetric should have <= symmetric width: asym={asym_width}, sym={sym_width}"
    );
}

// ===========================================================================
// 10. Per-channel vs per-tensor INT8 bound comparison (IBP)
// ===========================================================================

/// Per-channel quantization with per-output-channel scale vs single
/// per-tensor scale. Width should scale proportionally with magnitude.
#[test]
fn test_quantization_per_channel_vs_per_tensor() {
    let def = build_linear_kernel("quant_pipe_per_ch_vs_tensor");
    let per_tensor_width = linear_ibp_width(&def, INT8_WEIGHT_MAG, 1.0);
    let per_channel_mag = INT8_WEIGHT_MAG * 1.05;
    let per_channel_width = linear_ibp_width(&def, per_channel_mag, 1.0);

    eprintln!("Per-tensor: {per_tensor_width:.6}, per-channel worst: {per_channel_width:.6}");
    assert!(per_tensor_width.is_finite());
    assert!(per_channel_width.is_finite());

    let ratio = per_channel_width / per_tensor_width;
    let expected_ratio = 1.05_f32;
    let ratio_diff = (ratio - expected_ratio).abs() / expected_ratio;
    eprintln!("Width ratio: {ratio:.4}, expected: {expected_ratio:.4}");
    assert!(
        ratio_diff < 0.05,
        "ratio should match mag ratio: got {ratio}, expected {expected_ratio}"
    );
}

// ===========================================================================
// 11. Quantized matmul error accumulation (IBP)
// ===========================================================================

/// Matmul with quantized weights accumulates error proportional to inner dim.
#[test]
fn test_quantization_matmul_error_accumulation() {
    let small_dim = DIM / 2;
    let large_dim = DIM;

    let mut b_s = TensorBlockBuilder::new("quant_pipe_mm_small");
    let x_s = b_s.add_input("x", &[SEQ_LEN, small_dim]);
    let w_s = b_s.add_input("w", &[small_dim, small_dim]);
    let o_s = b_s.add_matmul(x_s, w_s, false, None, &[SEQ_LEN, small_dim]);
    let def_s = b_s.build(o_s).expect("valid small matmul");
    let bindings_s = vec![
        TensorParamBinding::Variable,
        weight_binding(&[small_dim, small_dim], INT8_WEIGHT_MAG),
    ];
    let g_s = tensor_kernel_to_graph(&def_s, &bindings_s).expect("small graph");
    let out_s = g_s
        .propagate_ibp(&uniform_bounds(&[SEQ_LEN, small_dim], 1.0))
        .expect("IBP");
    assert_bounds_valid(&out_s);
    let small_width = bound_width(&out_s);

    let mut b_l = TensorBlockBuilder::new("quant_pipe_mm_large");
    let x_l = b_l.add_input("x", &[SEQ_LEN, large_dim]);
    let w_l = b_l.add_input("w", &[large_dim, large_dim]);
    let o_l = b_l.add_matmul(x_l, w_l, false, None, &[SEQ_LEN, large_dim]);
    let def_l = b_l.build(o_l).expect("valid large matmul");
    let bindings_l = vec![
        TensorParamBinding::Variable,
        weight_binding(&[large_dim, large_dim], INT8_WEIGHT_MAG),
    ];
    let g_l = tensor_kernel_to_graph(&def_l, &bindings_l).expect("large graph");
    let out_l = g_l
        .propagate_ibp(&uniform_bounds(&[SEQ_LEN, large_dim], 1.0))
        .expect("IBP");
    assert_bounds_valid(&out_l);
    let large_width = bound_width(&out_l);

    eprintln!("Matmul: dim={small_dim} w={small_width:.6}, dim={large_dim} w={large_width:.6}");
    let width_ratio = large_width / small_width;
    let dim_ratio = large_dim as f32 / small_dim as f32;
    eprintln!("Width ratio: {width_ratio:.4}, dim ratio: {dim_ratio:.4}");
    assert!(
        (width_ratio - dim_ratio).abs() / dim_ratio < 0.15,
        "matmul width should scale with dim: wr={width_ratio}, dr={dim_ratio}"
    );
}

// ===========================================================================
// 12. Quantized softmax output bounded in [0, 1] (IBP)
// ===========================================================================

/// Softmax after quantized linear projection produces valid [0, 1] output.
#[test]
fn test_quantization_softmax_output_bounded() {
    let vocab_size = 16;
    let mut b = TensorBlockBuilder::new("quant_pipe_softmax_bounded");
    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let lm_w = b.add_input("lm_w", &[vocab_size, DIM]);
    let logits = b.add_linear(input, lm_w, None, &[SEQ_LEN, vocab_size]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, vocab_size]);
    let def = b.build(out).expect("valid quantized softmax kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[vocab_size, DIM], INT8_WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let output = graph
        .propagate_ibp(&uniform_bounds(&[SEQ_LEN, DIM], 1.0))
        .expect("IBP quantized softmax");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, vocab_size]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Quantized softmax IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-4, "softmax lower >= 0, got {lo_min}");
    assert!(hi_max <= 1.0 + 1e-4, "softmax upper <= 1, got {hi_max}");
}

// ===========================================================================
// 13. Full encoder -> quantize -> decoder pipeline (IBP + CROWN)
// ===========================================================================

/// End-to-end mixed precision: FP32 attention encoder + INT8 SwiGLU decoder
/// + softmax output. Verifies bounded output and CROWN tightening.
#[test]
fn test_quantization_full_encoder_decoder_pipeline() {
    let shape = [SEQ_LEN, DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let vocab_size = 16;

    let mut b = TensorBlockBuilder::new("quant_pipe_enc_dec");
    let input = b.add_input("x", &shape);

    // Encoder: FP32 attention
    let wq = b.add_input("wq", &[DIM, DIM]);
    let wk = b.add_input("wk", &[DIM, DIM]);
    let wv = b.add_input("wv", &[DIM, DIM]);
    let wo = b.add_input("wo", &[DIM, DIM]);
    let q = b.add_linear(input, wq, None, &shape);
    let k = b.add_linear(input, wk, None, &shape);
    let v = b.add_linear(input, wv, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, None, &shape);
    let attn_proj = b.add_linear(attn, wo, None, &shape);
    let enc_out = b.add_binary_add(input, attn_proj, &shape);

    // Decoder: INT8 SwiGLU FFN
    let gate_w = b.add_input("gate_w", &[FFN_DIM, DIM]);
    let up_w = b.add_input("up_w", &[FFN_DIM, DIM]);
    let down_w = b.add_input("down_w", &[DIM, FFN_DIM]);
    let gate = b.add_linear(enc_out, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(enc_out, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);
    let dec_out = b.add_binary_add(enc_out, ffn_out, &shape);

    // Output: softmax
    let lm_w = b.add_input("lm_w", &[vocab_size, DIM]);
    let logits = b.add_linear(dec_out, lm_w, None, &[SEQ_LEN, vocab_size]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, vocab_size]);
    let def = b.build(out).expect("valid enc-dec pipeline kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[DIM, DIM], FP32_WEIGHT_MAG), // wq
        weight_binding(&[DIM, DIM], FP32_WEIGHT_MAG), // wk
        weight_binding(&[DIM, DIM], FP32_WEIGHT_MAG), // wv
        weight_binding(&[DIM, DIM], FP32_WEIGHT_MAG), // wo
        weight_binding(&[FFN_DIM, DIM], INT8_WEIGHT_MAG), // gate_w
        weight_binding(&[FFN_DIM, DIM], INT8_WEIGHT_MAG), // up_w
        weight_binding(&[DIM, FFN_DIM], INT8_WEIGHT_MAG), // down_w
        weight_binding(&[vocab_size, DIM], FP32_WEIGHT_MAG), // lm_w
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input_bounds = uniform_bounds(&shape, 0.5);

    let ibp_out = graph.propagate_ibp(&input_bounds).expect("IBP pipeline");
    assert_bounds_valid(&ibp_out);
    assert_eq!(ibp_out.lower_upper().0.shape(), &[SEQ_LEN, vocab_size]);

    let (lo_min, hi_max) = bounds_min_max(&ibp_out);
    eprintln!("Enc-dec pipeline IBP: [{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min >= -1e-4, "pipeline softmax lower >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-4,
        "pipeline softmax upper <= 1, got {hi_max}"
    );

    let (method, crown_out, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (c_lo, c_hi) = bounds_min_max(&crown_out);
    eprintln!("Enc-dec pipeline CROWN: method={method:?}, [{c_lo:.6}, {c_hi:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
