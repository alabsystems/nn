// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: INT8 W8A16 quantization NY soundness proof.
//!
//! Proves that INT8 weight-only quantization preserves output bounds for
//! Linear layers using NY IBP and CROWN propagation.
//!
//! Two quantization schemes tested:
//!
//! 1. **Symmetric** (W8A16): `w_f32 = w_i8 * scale`, where
//!    `scale = max(|w_row|) / 127`. No zero_point.
//!
//! 2. **Asymmetric** (W8A16 with zero_point): `w_f32 = (w_i8 - zp) * scale`,
//!    where `scale = (w_max - w_min) / 255` and `zp = round(-w_min / scale)`.
//!    The zero_point shift introduces additional perturbation vs symmetric.
//!
//! Strategy:
//! 1. Build Linear layer with F32 weights via TensorBlockBuilder.
//! 2. Build identical Linear layer with INT8-quantized weights
//!    (roundtripped: f32 -> int8 -> f32 to simulate quantization error).
//! 3. Propagate input bounds [-10, 10] (typical ViT range) through both.
//! 4. Verify both produce finite, valid bounds.
//! 5. Verify quantization drift is bounded by epsilon.
//! 6. Use IbpValidated soundness mode (via Conservative NormBoundsMode).
//!
//! Part of #3533.
//! Part of #3525.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert, verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding, VerifyConfig};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

/// Small layer: 64 -> 64 (fast tests, validates basic quantization path).
const SMALL_IN: usize = 64;
const SMALL_OUT: usize = 64;
const SMALL_SEQ: usize = 8;

/// Medium layer: 256 -> 256 (realistic ViT-like dimensions).
const MED_IN: usize = 256;
const MED_OUT: usize = 256;
const MED_SEQ: usize = 4;

/// Typical ViT activation range for input bounds.
const VIT_RANGE: f32 = 10.0;

// ---------------------------------------------------------------------------
// INT8 per-channel symmetric quantization (W8A16)
// ---------------------------------------------------------------------------

/// Simulate INT8 per-channel symmetric quantization of a weight matrix.
///
/// For each output channel (row), computes:
///   scale = max(|w_row|) / 127
///   w_i8 = round(w / scale), clamped to [-128, 127]
///   w_deq = w_i8 * scale
///
/// Returns the dequantized f32 weights (same shape, but only values
/// representable in INT8 at the per-channel scale).
fn quantize_int8_symmetric(weights: &ArrayD<f32>) -> ArrayD<f32> {
    let shape = weights.shape();
    assert!(shape.len() == 2, "expected 2D weight matrix");
    let (out_ch, in_ch) = (shape[0], shape[1]);

    let mut result = weights.clone();
    for oc in 0..out_ch {
        // Per-channel scale: max absolute value in this row / 127
        let row_max = (0..in_ch)
            .map(|ic| weights[[oc, ic]].abs())
            .fold(0.0f32, f32::max);

        if row_max == 0.0 {
            // Zero row stays zero.
            continue;
        }

        let scale = row_max / 127.0;
        for ic in 0..in_ch {
            let w = weights[[oc, ic]];
            let w_i8 = (w / scale).round().clamp(-128.0, 127.0);
            result[[oc, ic]] = w_i8 * scale;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// INT8 per-channel asymmetric quantization (W8A16 with zero_point)
// ---------------------------------------------------------------------------

/// Simulate INT8 per-channel asymmetric quantization of a weight matrix.
///
/// For each output channel (row), computes (standard affine scheme):
///   w_min = min(min(w_row), 0), w_max = max(max(w_row), 0)
///   scale = (w_max - w_min) / 255
///   zero_point = round(-w_min / scale), clamped to [0, 255]
///   w_u8 = round(w / scale) + zero_point, clamped to [0, 255]
///   w_deq = (w_u8 - zero_point) * scale
///
/// Asymmetric quantization uses the full [0, 255] range and introduces
/// a zero_point offset. This models the additional perturbation from
/// zero_point rounding that symmetric quantization avoids.
///
/// IMPORTANT: the quantization range MUST include the real value 0
/// (`min(.., 0)` / `max(.., 0)`). This is mandatory for affine/asymmetric
/// quantization: the quantized grid `[zp, zp + 255] * scale` covers
/// `[w_min, w_max]`, and only if 0 lies inside that range does a valid
/// `zero_point in [0, 255]` exist that keeps every weight representable.
/// If the range is taken over the data alone (e.g. an all-positive row),
/// the zero_point clamps to 0 and the grid no longer covers the data,
/// producing per-element error up to the full range instead of scale/2.
///
/// Returns the dequantized f32 weights.
fn quantize_int8_asymmetric(weights: &ArrayD<f32>) -> ArrayD<f32> {
    let shape = weights.shape();
    assert!(shape.len() == 2, "expected 2D weight matrix");
    let (out_ch, in_ch) = (shape[0], shape[1]);

    let mut result = weights.clone();
    for oc in 0..out_ch {
        let row_min = (0..in_ch)
            .map(|ic| weights[[oc, ic]])
            .fold(f32::INFINITY, f32::min)
            .min(0.0);
        let row_max = (0..in_ch)
            .map(|ic| weights[[oc, ic]])
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0);

        let range = row_max - row_min;
        if range == 0.0 {
            // Constant row: all values map to the same quantized level.
            continue;
        }

        let scale = range / 255.0;
        let zero_point = (-row_min / scale).round().clamp(0.0, 255.0);

        for ic in 0..in_ch {
            let w = weights[[oc, ic]];
            let w_u8 = (w / scale + zero_point).round().clamp(0.0, 255.0);
            result[[oc, ic]] = (w_u8 - zero_point) * scale;
        }
    }
    result
}

/// Compute per-channel asymmetric quantization scales and zero_points.
///
/// Returns `(scales, zero_points)` where each vector has `out_channels` elements.
/// Used to verify zero_point perturbation bounds analytically.
fn asymmetric_scales_and_zero_points(weights: &ArrayD<f32>) -> (Vec<f32>, Vec<f32>) {
    let shape = weights.shape();
    let (out_ch, in_ch) = (shape[0], shape[1]);
    let mut scales = Vec::with_capacity(out_ch);
    let mut zero_points = Vec::with_capacity(out_ch);

    for oc in 0..out_ch {
        // Affine asymmetric range must include the real value 0 (see
        // `quantize_int8_asymmetric`) so a valid zero_point in [0, 255] exists.
        let row_min = (0..in_ch)
            .map(|ic| weights[[oc, ic]])
            .fold(f32::INFINITY, f32::min)
            .min(0.0);
        let row_max = (0..in_ch)
            .map(|ic| weights[[oc, ic]])
            .fold(f32::NEG_INFINITY, f32::max)
            .max(0.0);

        let range = row_max - row_min;
        if range == 0.0 {
            scales.push(0.0);
            zero_points.push(0.0);
            continue;
        }

        let scale = range / 255.0;
        let zp = (-row_min / scale).round().clamp(0.0, 255.0);
        scales.push(scale);
        zero_points.push(zp);
    }

    (scales, zero_points)
}

// ---------------------------------------------------------------------------
// IbpValidated soundness config (Conservative NormBoundsMode)
// ---------------------------------------------------------------------------

/// VerifyConfig using Conservative NormBoundsMode which maps to
/// `LayerNormCrownMode::IbpValidated` -- Jacobian-based CROWN linearization
/// with IBP-validated error margins. Provably sound.
fn ibp_validated_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build a single Linear layer kernel: input @ weight^T + bias.
///
/// Input: `[seq_len, in_features]` (Variable).
/// Output: `[seq_len, out_features]`.
fn build_linear_kernel(
    name: &str,
    seq_len: usize,
    in_features: usize,
    out_features: usize,
) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);

    let input = b.add_input("x", &[seq_len, in_features]);
    let weight = b.add_input("weight", &[out_features, in_features]);
    let bias = b.add_input("bias", &[out_features]);

    let out = b.add_linear(input, weight, Some(bias), &[seq_len, out_features]);
    b.build(out).expect("valid Linear kernel")
}

/// Create bindings for a Linear layer with given weight values.
///
/// Input is Variable, weight and bias are ConstantTensor.
fn linear_bindings(weights: &ArrayD<f32>, bias: &ArrayD<f32>) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,                        // x
        TensorParamBinding::ConstantTensor(weights.clone()), // weight
        TensorParamBinding::ConstantTensor(bias.clone()),    // bias
    ]
}

/// Generate Xavier-normal-like weights for a given shape.
///
/// Uses deterministic pseudo-random values scaled by sqrt(2 / (fan_in + fan_out))
/// to produce realistic weight magnitudes. The pattern is deterministic for
/// reproducibility.
fn xavier_weights(out_features: usize, in_features: usize) -> ArrayD<f32> {
    let fan_sum = (in_features + out_features) as f32;
    let std_dev = (2.0 / fan_sum).sqrt();
    let n = out_features * in_features;
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        // Deterministic pseudo-random using a simple hash-like pattern.
        // Produces values in roughly [-std_dev, +std_dev].
        let t = i as f32 / n as f32;
        let val = (t * 7.37 + 0.13).sin() * std_dev;
        data.push(val);
    }
    ArrayD::from_shape_vec(IxDyn(&[out_features, in_features]), data).expect("valid weight shape")
}

// ---------------------------------------------------------------------------
// Tests: small layer (64 -> 64)
// ---------------------------------------------------------------------------

/// INT8 Linear kernel definition validates (small layer).
#[test]
fn test_int8_small_linear_def_validates() {
    let def = build_linear_kernel("int8_small_linear", SMALL_SEQ, SMALL_IN, SMALL_OUT);
    def.validate().expect("Small Linear kernel should validate");
}

/// INT8 Linear translates to NY GraphNetwork (small layer).
#[test]
fn test_int8_small_linear_graph_builds() {
    let def = build_linear_kernel("int8_small_linear", SMALL_SEQ, SMALL_IN, SMALL_OUT);
    let weights = xavier_weights(SMALL_OUT, SMALL_IN);
    let bias = ArrayD::from_elem(IxDyn(&[SMALL_OUT]), 0.0f32);
    let bindings = linear_bindings(&weights, &bias);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("Linear graph should translate");
    assert!(
        graph.num_nodes() >= 1,
        "Linear graph should have >= 1 node, got {}",
        graph.num_nodes()
    );
}

/// F32 IBP bounds propagate through small Linear layer.
#[test]
fn test_int8_small_f32_ibp_propagates() {
    let def = build_linear_kernel("int8_small_linear", SMALL_SEQ, SMALL_IN, SMALL_OUT);
    let weights = xavier_weights(SMALL_OUT, SMALL_IN);
    let bias = ArrayD::from_elem(IxDyn(&[SMALL_OUT]), 0.0f32);
    let bindings = linear_bindings(&weights, &bias);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SMALL_SEQ, SMALL_IN], VIT_RANGE);

    let output = graph.propagate_ibp(&input).expect("IBP through Linear");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SMALL_SEQ, SMALL_OUT],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Small F32 Linear IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// INT8-quantized IBP bounds propagate through small Linear layer.
#[test]
fn test_int8_small_quantized_ibp_propagates() {
    let def = build_linear_kernel("int8_small_linear_q", SMALL_SEQ, SMALL_IN, SMALL_OUT);
    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let weights_int8 = quantize_int8_symmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[SMALL_OUT]), 0.0f32);
    let bindings = linear_bindings(&weights_int8, &bias);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SMALL_SEQ, SMALL_IN], VIT_RANGE);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantized Linear");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SMALL_SEQ, SMALL_OUT],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Small INT8 Linear IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// INT8 quantization drift is bounded for small Linear layer.
///
/// The key soundness property: the difference between F32 and INT8
/// output bounds is bounded. For W8A16 with per-channel symmetric
/// quantization, the max weight error per element is scale/2 where
/// scale = max(|w_row|)/127. For Xavier-initialized weights with
/// in_features=64, this gives ~0.001 per weight element, and the
/// accumulated output error across in_features=64 is bounded.
#[test]
fn test_int8_small_quantization_drift_bounded() {
    let def_f32 = build_linear_kernel("int8_drift_f32", SMALL_SEQ, SMALL_IN, SMALL_OUT);
    let def_q = build_linear_kernel("int8_drift_q", SMALL_SEQ, SMALL_IN, SMALL_OUT);

    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let weights_int8 = quantize_int8_symmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[SMALL_OUT]), 0.0f32);

    let bindings_f32 = linear_bindings(&weights_f32, &bias);
    let bindings_q = linear_bindings(&weights_int8, &bias);

    let graph_f32 = tensor_kernel_to_graph(&def_f32, &bindings_f32).expect("f32 graph");
    let graph_q = tensor_kernel_to_graph(&def_q, &bindings_q).expect("int8 graph");

    let input = uniform_bounds(&[SMALL_SEQ, SMALL_IN], VIT_RANGE);

    let out_f32 = graph_f32.propagate_ibp(&input).expect("f32 IBP");
    let out_q = graph_q.propagate_ibp(&input).expect("int8 IBP");

    assert_bounds_valid(&out_f32);
    assert_bounds_valid(&out_q);

    let (f32_lo, f32_hi) = out_f32.lower_upper();
    let (q_lo, q_hi) = out_q.lower_upper();

    // Compute max absolute drift between F32 and INT8 bounds.
    let max_lo_drift = f32_lo
        .iter()
        .zip(q_lo.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_hi_drift = f32_hi
        .iter()
        .zip(q_hi.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let max_drift = max_lo_drift.max(max_hi_drift);
    eprintln!(
        "Small INT8 drift: max_lo_drift={max_lo_drift:.6}, max_hi_drift={max_hi_drift:.6}, \
         max_drift={max_drift:.6}"
    );

    // INT8 per-channel symmetric quantization on Xavier weights with in=64:
    // max_weight_error ≈ scale/2 ≈ max(|w|)/(2*127) ≈ 0.18/(254) ≈ 0.0007
    // Accumulated over 64 input features with input range [-10, 10]:
    // max_output_drift ≈ in_features * input_range * max_weight_error
    //                   ≈ 64 * 10 * 0.0007 ≈ 0.45
    // Use 5.0 as a conservative upper bound.
    assert!(
        max_drift < 5.0,
        "INT8 quantization drift should be < 5.0, got {max_drift}"
    );
    assert!(max_drift.is_finite(), "drift must be finite");
}

/// CROWN bounds propagate through small INT8-quantized Linear layer.
#[test]
fn test_int8_small_crown_propagation() {
    let def = build_linear_kernel("int8_small_crown", SMALL_SEQ, SMALL_IN, SMALL_OUT);
    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let weights_int8 = quantize_int8_symmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[SMALL_OUT]), 0.0f32);
    let bindings = linear_bindings(&weights_int8, &bias);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SMALL_SEQ, SMALL_IN], VIT_RANGE);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SMALL_SEQ, SMALL_OUT],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Small INT8 CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record small INT8 quantized Linear under status key.
#[test]
fn test_int8_small_verify_and_record() {
    let def = build_linear_kernel("int8_small_linear_verify", SMALL_SEQ, SMALL_IN, SMALL_OUT);
    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let weights_int8 = quantize_int8_symmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[SMALL_OUT]), 0.0f32);
    let bindings = linear_bindings(&weights_int8, &bias);
    let input = uniform_bounds(&[SMALL_SEQ, SMALL_IN], VIT_RANGE);

    let result = verify_and_assert(&def, &bindings, &input, "int8_linear_small");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SMALL_SEQ, SMALL_OUT]);
}

// ---------------------------------------------------------------------------
// Tests: medium layer (256 -> 256)
// ---------------------------------------------------------------------------

/// INT8-quantized IBP bounds propagate through medium Linear layer.
#[test]
fn test_int8_medium_quantized_ibp_propagates() {
    let def = build_linear_kernel("int8_med_linear_q", MED_SEQ, MED_IN, MED_OUT);
    let weights_f32 = xavier_weights(MED_OUT, MED_IN);
    let weights_int8 = quantize_int8_symmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[MED_OUT]), 0.0f32);
    let bindings = linear_bindings(&weights_int8, &bias);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[MED_SEQ, MED_IN], VIT_RANGE);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantized Linear");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[MED_SEQ, MED_OUT],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Medium INT8 Linear IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// INT8 quantization drift is bounded for medium Linear layer.
#[test]
fn test_int8_medium_quantization_drift_bounded() {
    let def_f32 = build_linear_kernel("int8_med_drift_f32", MED_SEQ, MED_IN, MED_OUT);
    let def_q = build_linear_kernel("int8_med_drift_q", MED_SEQ, MED_IN, MED_OUT);

    let weights_f32 = xavier_weights(MED_OUT, MED_IN);
    let weights_int8 = quantize_int8_symmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[MED_OUT]), 0.0f32);

    let bindings_f32 = linear_bindings(&weights_f32, &bias);
    let bindings_q = linear_bindings(&weights_int8, &bias);

    let graph_f32 = tensor_kernel_to_graph(&def_f32, &bindings_f32).expect("f32 graph");
    let graph_q = tensor_kernel_to_graph(&def_q, &bindings_q).expect("int8 graph");

    let input = uniform_bounds(&[MED_SEQ, MED_IN], VIT_RANGE);

    let out_f32 = graph_f32.propagate_ibp(&input).expect("f32 IBP");
    let out_q = graph_q.propagate_ibp(&input).expect("int8 IBP");

    assert_bounds_valid(&out_f32);
    assert_bounds_valid(&out_q);

    let (f32_lo, f32_hi) = out_f32.lower_upper();
    let (q_lo, q_hi) = out_q.lower_upper();

    let max_lo_drift = f32_lo
        .iter()
        .zip(q_lo.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_hi_drift = f32_hi
        .iter()
        .zip(q_hi.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let max_drift = max_lo_drift.max(max_hi_drift);
    eprintln!(
        "Medium INT8 drift: max_lo_drift={max_lo_drift:.6}, max_hi_drift={max_hi_drift:.6}, \
         max_drift={max_drift:.6}"
    );

    // Medium layer (256 features) accumulates more error but is still bounded.
    // max_output_drift ≈ in_features * input_range * max_weight_error
    //                   ≈ 256 * 10 * 0.0004 ≈ 1.0
    // Use 10.0 as a conservative upper bound.
    assert!(
        max_drift < 10.0,
        "INT8 quantization drift should be < 10.0, got {max_drift}"
    );
    assert!(max_drift.is_finite(), "drift must be finite");
}

/// CROWN bounds propagate through medium INT8-quantized Linear layer.
#[test]
fn test_int8_medium_crown_propagation() {
    let def = build_linear_kernel("int8_med_crown", MED_SEQ, MED_IN, MED_OUT);
    let weights_f32 = xavier_weights(MED_OUT, MED_IN);
    let weights_int8 = quantize_int8_symmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[MED_OUT]), 0.0f32);
    let bindings = linear_bindings(&weights_int8, &bias);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[MED_SEQ, MED_IN], VIT_RANGE);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[MED_SEQ, MED_OUT],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Medium INT8 CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// Verify and record medium INT8 quantized Linear under status key.
#[test]
fn test_int8_medium_verify_and_record() {
    let def = build_linear_kernel("int8_med_linear_verify", MED_SEQ, MED_IN, MED_OUT);
    let weights_f32 = xavier_weights(MED_OUT, MED_IN);
    let weights_int8 = quantize_int8_symmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[MED_OUT]), 0.0f32);
    let bindings = linear_bindings(&weights_int8, &bias);
    let input = uniform_bounds(&[MED_SEQ, MED_IN], VIT_RANGE);

    let result = verify_and_assert(&def, &bindings, &input, "int8_linear_medium");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[MED_SEQ, MED_OUT]);
}

// ---------------------------------------------------------------------------
// Tests: quantization error bounds (analytical)
// ---------------------------------------------------------------------------

/// Verify INT8 quantization error is within theoretical maximum.
///
/// For symmetric per-channel INT8 quantization:
///   scale = max(|w_row|) / 127
///   max_error_per_element = scale / 2
///
/// This is the fundamental property that makes W8A16 verification possible:
/// the quantization error is deterministically bounded.
#[test]
fn test_int8_quantization_error_within_theoretical_max() {
    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let weights_int8 = quantize_int8_symmetric(&weights_f32);

    let shape = weights_f32.shape();
    let (out_ch, in_ch) = (shape[0], shape[1]);

    for oc in 0..out_ch {
        let row_max = (0..in_ch)
            .map(|ic| weights_f32[[oc, ic]].abs())
            .fold(0.0f32, f32::max);

        if row_max == 0.0 {
            continue;
        }

        let scale = row_max / 127.0;
        let theoretical_max_error = scale / 2.0;

        for ic in 0..in_ch {
            let error = (weights_f32[[oc, ic]] - weights_int8[[oc, ic]]).abs();
            assert!(
                error <= theoretical_max_error + 1e-7,
                "Channel {oc}, element {ic}: error {error:.8} exceeds \
                 theoretical max {theoretical_max_error:.8} (scale={scale:.8})"
            );
        }
    }
}

// ===========================================================================
// Tests: asymmetric quantization with zero_point perturbation
// ===========================================================================

/// Asymmetric INT8 quantization error is within theoretical maximum.
///
/// For asymmetric per-channel INT8 quantization:
///   scale = (w_max - w_min) / 255
///   max_error_per_element = scale / 2  (same bound as symmetric)
///
/// But zero_point rounding introduces an additional systematic shift
/// per channel. This test verifies the per-element error bound holds
/// for the asymmetric scheme as well.
#[test]
fn test_int8_asymmetric_quantization_error_bounded() {
    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let weights_asym = quantize_int8_asymmetric(&weights_f32);
    let (scales, _zero_points) = asymmetric_scales_and_zero_points(&weights_f32);

    let shape = weights_f32.shape();
    let (out_ch, in_ch) = (shape[0], shape[1]);

    for oc in 0..out_ch {
        let scale = scales[oc];
        if scale == 0.0 {
            continue;
        }

        // Theoretical max error per element for asymmetric quantization:
        // scale / 2 (rounding error) is the dominant term.
        let theoretical_max_error = scale / 2.0;

        for ic in 0..in_ch {
            let error = (weights_f32[[oc, ic]] - weights_asym[[oc, ic]]).abs();
            assert!(
                error <= theoretical_max_error + 1e-6,
                "Asymmetric channel {oc}, element {ic}: error {error:.8} exceeds \
                 theoretical max {theoretical_max_error:.8} (scale={scale:.8})"
            );
        }
    }
}

/// Zero_point values are valid integers in [0, 255] for all channels.
#[test]
fn test_int8_asymmetric_zero_points_valid() {
    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let (scales, zero_points) = asymmetric_scales_and_zero_points(&weights_f32);

    for (oc, (&scale, &zp)) in scales.iter().zip(zero_points.iter()).enumerate() {
        if scale == 0.0 {
            assert_eq!(zp, 0.0, "Channel {oc}: zero scale should have zp=0");
            continue;
        }
        assert!(
            (0.0..=255.0).contains(&zp),
            "Channel {oc}: zero_point {zp} out of [0, 255] range"
        );
        assert_eq!(
            zp,
            zp.round(),
            "Channel {oc}: zero_point {zp} should be integer"
        );
    }
}

/// Asymmetric INT8 quantized IBP bounds propagate through small Linear layer.
#[test]
fn test_int8_small_asymmetric_ibp_propagates() {
    let def = build_linear_kernel("int8_small_asym_q", SMALL_SEQ, SMALL_IN, SMALL_OUT);
    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let weights_asym = quantize_int8_asymmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[SMALL_OUT]), 0.0f32);
    let bindings = linear_bindings(&weights_asym, &bias);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[SMALL_SEQ, SMALL_IN], VIT_RANGE);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through asymmetric quantized Linear");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SMALL_SEQ, SMALL_OUT],
        "output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Small asymmetric INT8 Linear IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Asymmetric INT8 quantization drift (with zero_point) is bounded for small layer.
///
/// Asymmetric quantization introduces zero_point perturbation on top of
/// the rounding error. The drift should still be bounded but may be
/// slightly larger than symmetric quantization.
#[test]
fn test_int8_small_asymmetric_drift_bounded() {
    let def_f32 = build_linear_kernel("int8_asym_drift_f32", SMALL_SEQ, SMALL_IN, SMALL_OUT);
    let def_q = build_linear_kernel("int8_asym_drift_q", SMALL_SEQ, SMALL_IN, SMALL_OUT);

    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let weights_asym = quantize_int8_asymmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[SMALL_OUT]), 0.0f32);

    let bindings_f32 = linear_bindings(&weights_f32, &bias);
    let bindings_q = linear_bindings(&weights_asym, &bias);

    let graph_f32 = tensor_kernel_to_graph(&def_f32, &bindings_f32).expect("f32 graph");
    let graph_q = tensor_kernel_to_graph(&def_q, &bindings_q).expect("int8 graph");

    let input = uniform_bounds(&[SMALL_SEQ, SMALL_IN], VIT_RANGE);

    let out_f32 = graph_f32.propagate_ibp(&input).expect("f32 IBP");
    let out_q = graph_q.propagate_ibp(&input).expect("asymmetric int8 IBP");

    assert_bounds_valid(&out_f32);
    assert_bounds_valid(&out_q);

    let (f32_lo, f32_hi) = out_f32.lower_upper();
    let (q_lo, q_hi) = out_q.lower_upper();

    let max_lo_drift = f32_lo
        .iter()
        .zip(q_lo.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_hi_drift = f32_hi
        .iter()
        .zip(q_hi.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let max_drift = max_lo_drift.max(max_hi_drift);
    eprintln!(
        "Small asymmetric INT8 drift: max_lo={max_lo_drift:.6}, max_hi={max_hi_drift:.6}, \
         max={max_drift:.6}"
    );

    // Asymmetric quantization has similar per-element error bound (scale/2)
    // but zero_point rounding adds a small systematic shift per channel.
    // Use same conservative bound as symmetric: 5.0.
    assert!(
        max_drift < 5.0,
        "Asymmetric INT8 quantization drift should be < 5.0, got {max_drift}"
    );
    assert!(max_drift.is_finite(), "drift must be finite");
}

/// Asymmetric INT8 drift for medium layer (256 -> 256) is bounded.
#[test]
fn test_int8_medium_asymmetric_drift_bounded() {
    let def_f32 = build_linear_kernel("int8_med_asym_drift_f32", MED_SEQ, MED_IN, MED_OUT);
    let def_q = build_linear_kernel("int8_med_asym_drift_q", MED_SEQ, MED_IN, MED_OUT);

    let weights_f32 = xavier_weights(MED_OUT, MED_IN);
    let weights_asym = quantize_int8_asymmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[MED_OUT]), 0.0f32);

    let bindings_f32 = linear_bindings(&weights_f32, &bias);
    let bindings_q = linear_bindings(&weights_asym, &bias);

    let graph_f32 = tensor_kernel_to_graph(&def_f32, &bindings_f32).expect("f32 graph");
    let graph_q = tensor_kernel_to_graph(&def_q, &bindings_q).expect("int8 graph");

    let input = uniform_bounds(&[MED_SEQ, MED_IN], VIT_RANGE);

    let out_f32 = graph_f32.propagate_ibp(&input).expect("f32 IBP");
    let out_q = graph_q.propagate_ibp(&input).expect("asymmetric int8 IBP");

    assert_bounds_valid(&out_f32);
    assert_bounds_valid(&out_q);

    let (f32_lo, f32_hi) = out_f32.lower_upper();
    let (q_lo, q_hi) = out_q.lower_upper();

    let max_lo_drift = f32_lo
        .iter()
        .zip(q_lo.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_hi_drift = f32_hi
        .iter()
        .zip(q_hi.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let max_drift = max_lo_drift.max(max_hi_drift);
    eprintln!(
        "Medium asymmetric INT8 drift: max_lo={max_lo_drift:.6}, max_hi={max_hi_drift:.6}, \
         max={max_drift:.6}"
    );

    // Medium layer (256 features) accumulates more error.
    assert!(
        max_drift < 10.0,
        "Asymmetric INT8 quantization drift should be < 10.0, got {max_drift}"
    );
    assert!(max_drift.is_finite(), "drift must be finite");
}

/// Symmetric vs asymmetric drift comparison: asymmetric should not be
/// dramatically worse than symmetric for Xavier-initialized weights.
#[test]
fn test_int8_symmetric_vs_asymmetric_drift_comparable() {
    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let weights_sym = quantize_int8_symmetric(&weights_f32);
    let weights_asym = quantize_int8_asymmetric(&weights_f32);

    let shape = weights_f32.shape();
    let n = shape[0] * shape[1];

    let sym_mse: f32 = weights_f32
        .iter()
        .zip(weights_sym.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / n as f32;

    let asym_mse: f32 = weights_f32
        .iter()
        .zip(weights_asym.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        / n as f32;

    eprintln!(
        "Symmetric MSE: {sym_mse:.8}, Asymmetric MSE: {asym_mse:.8}, ratio: {:.2}",
        asym_mse / sym_mse.max(1e-12)
    );

    // For Xavier weights (roughly symmetric distribution), asymmetric
    // quantization should have comparable or slightly larger MSE.
    // Allow up to 4x ratio since asymmetric uses [0,255] vs [-128,127].
    assert!(
        asym_mse < sym_mse * 4.0 + 1e-10,
        "Asymmetric MSE {asym_mse:.8} should not be dramatically worse than \
         symmetric MSE {sym_mse:.8}"
    );
}

// ===========================================================================
// Tests: IbpValidated soundness mode verification
// ===========================================================================

/// Verify small INT8 quantized Linear with IbpValidated soundness mode.
///
/// Uses Conservative NormBoundsMode which maps to IbpValidated crown mode.
/// For a pure Linear layer (no normalization), the soundness mode should
/// be Sound since there are no heuristic approximations.
#[test]
fn test_int8_small_ibp_validated_verify() {
    let def = build_linear_kernel("int8_small_ibpval", SMALL_SEQ, SMALL_IN, SMALL_OUT);
    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let weights_int8 = quantize_int8_symmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[SMALL_OUT]), 0.0f32);
    let bindings = linear_bindings(&weights_int8, &bias);
    let input = uniform_bounds(&[SMALL_SEQ, SMALL_IN], VIT_RANGE);

    let config = ibp_validated_config();
    let result =
        verify_and_assert_with_config(&def, &bindings, &input, "int8_linear_small_ibpval", &config);

    assert_eq!(result.num_variables, 1, "single Variable input");

    // Pure Linear layer should verify as Sound (no normalization layers
    // that would trigger heuristic classification).
    eprintln!(
        "Small INT8 IbpValidated: soundness={:?}, width={:.4}",
        result.verification.soundness_mode, result.verification.output_width
    );
    assert!(
        result.verification.is_finite,
        "output bounds must be finite"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SMALL_SEQ, SMALL_OUT]);
}

/// Verify medium INT8 quantized Linear with IbpValidated soundness mode.
#[test]
fn test_int8_medium_ibp_validated_verify() {
    let def = build_linear_kernel("int8_med_ibpval", MED_SEQ, MED_IN, MED_OUT);
    let weights_f32 = xavier_weights(MED_OUT, MED_IN);
    let weights_int8 = quantize_int8_symmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[MED_OUT]), 0.0f32);
    let bindings = linear_bindings(&weights_int8, &bias);
    let input = uniform_bounds(&[MED_SEQ, MED_IN], VIT_RANGE);

    let config = ibp_validated_config();
    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "int8_linear_medium_ibpval",
        &config,
    );

    assert_eq!(result.num_variables, 1, "single Variable input");

    eprintln!(
        "Medium INT8 IbpValidated: soundness={:?}, width={:.4}",
        result.verification.soundness_mode, result.verification.output_width
    );
    assert!(
        result.verification.is_finite,
        "output bounds must be finite"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[MED_SEQ, MED_OUT]);
}

/// Verify small asymmetric INT8 quantized Linear with IbpValidated soundness.
#[test]
fn test_int8_small_asymmetric_ibp_validated_verify() {
    let def = build_linear_kernel("int8_small_asym_ibpval", SMALL_SEQ, SMALL_IN, SMALL_OUT);
    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let weights_asym = quantize_int8_asymmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[SMALL_OUT]), 0.0f32);
    let bindings = linear_bindings(&weights_asym, &bias);
    let input = uniform_bounds(&[SMALL_SEQ, SMALL_IN], VIT_RANGE);

    let config = ibp_validated_config();
    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "int8_linear_small_asym_ibpval",
        &config,
    );

    assert_eq!(result.num_variables, 1, "single Variable input");

    eprintln!(
        "Small asymmetric INT8 IbpValidated: soundness={:?}, width={:.4}",
        result.verification.soundness_mode, result.verification.output_width
    );
    assert!(
        result.verification.is_finite,
        "output bounds must be finite"
    );
}

/// Verify medium asymmetric INT8 quantized Linear with IbpValidated soundness.
#[test]
fn test_int8_medium_asymmetric_ibp_validated_verify() {
    let def = build_linear_kernel("int8_med_asym_ibpval", MED_SEQ, MED_IN, MED_OUT);
    let weights_f32 = xavier_weights(MED_OUT, MED_IN);
    let weights_asym = quantize_int8_asymmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[MED_OUT]), 0.0f32);
    let bindings = linear_bindings(&weights_asym, &bias);
    let input = uniform_bounds(&[MED_SEQ, MED_IN], VIT_RANGE);

    let config = ibp_validated_config();
    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "int8_linear_medium_asym_ibpval",
        &config,
    );

    assert_eq!(result.num_variables, 1, "single Variable input");

    eprintln!(
        "Medium asymmetric INT8 IbpValidated: soundness={:?}, width={:.4}",
        result.verification.soundness_mode, result.verification.output_width
    );
    assert!(
        result.verification.is_finite,
        "output bounds must be finite"
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[MED_SEQ, MED_OUT]);
}

// ===========================================================================
// Tests: quantization noise modeling (epsilon-bounded deviation proof)
// ===========================================================================

/// Prove output deviation from INT8 quantization is bounded by epsilon.
///
/// The core soundness property: for a Linear layer y = W @ x + b,
/// quantization replaces W with W_q = W + dW where |dW[i,j]| <= scale[i]/2.
///
/// The output deviation dy = dW @ x is bounded by:
///   |dy[i]| <= sum_j(|dW[i,j]| * |x[j]|) <= (scale[i]/2) * sum_j(|x[j]|)
///
/// For input bounds [-R, R] across all features:
///   |dy[i]| <= (scale[i]/2) * in_features * R
///
/// This test verifies the NY IBP-computed drift matches
/// the analytical epsilon bound.
#[test]
fn test_int8_small_epsilon_deviation_proof() {
    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let weights_int8 = quantize_int8_symmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[SMALL_OUT]), 0.0f32);

    // Compute analytical epsilon per output channel.
    let shape = weights_f32.shape();
    let (out_ch, in_ch) = (shape[0], shape[1]);
    let mut epsilon_per_channel = Vec::with_capacity(out_ch);
    for oc in 0..out_ch {
        let row_max = (0..in_ch)
            .map(|ic| weights_f32[[oc, ic]].abs())
            .fold(0.0f32, f32::max);
        if row_max == 0.0 {
            epsilon_per_channel.push(0.0);
            continue;
        }
        let scale = row_max / 127.0;
        // Worst-case output deviation for this channel:
        // (scale/2) * in_features * VIT_RANGE
        let eps = (scale / 2.0) * in_ch as f32 * VIT_RANGE;
        epsilon_per_channel.push(eps);
    }

    // Now verify via NY IBP that the actual drift is within epsilon.
    let def_f32 = build_linear_kernel("int8_eps_f32", SMALL_SEQ, SMALL_IN, SMALL_OUT);
    let def_q = build_linear_kernel("int8_eps_q", SMALL_SEQ, SMALL_IN, SMALL_OUT);

    let bindings_f32 = linear_bindings(&weights_f32, &bias);
    let bindings_q = linear_bindings(&weights_int8, &bias);

    let graph_f32 = tensor_kernel_to_graph(&def_f32, &bindings_f32).expect("f32 graph");
    let graph_q = tensor_kernel_to_graph(&def_q, &bindings_q).expect("int8 graph");

    let input = uniform_bounds(&[SMALL_SEQ, SMALL_IN], VIT_RANGE);

    let out_f32 = graph_f32.propagate_ibp(&input).expect("f32 IBP");
    let out_q = graph_q.propagate_ibp(&input).expect("int8 IBP");

    let (f32_lo, f32_hi) = out_f32.lower_upper();
    let (q_lo, q_hi) = out_q.lower_upper();

    // Check per-channel drift against analytical epsilon.
    for s in 0..SMALL_SEQ {
        for oc in 0..SMALL_OUT {
            let lo_drift = (f32_lo[[s, oc]] - q_lo[[s, oc]]).abs();
            let hi_drift = (f32_hi[[s, oc]] - q_hi[[s, oc]]).abs();
            let max_ch_drift = lo_drift.max(hi_drift);
            let eps = epsilon_per_channel[oc];

            // Allow small numerical tolerance on top of analytical bound.
            assert!(
                max_ch_drift <= eps + 1e-4,
                "seq={s}, ch={oc}: drift {max_ch_drift:.6} exceeds analytical \
                 epsilon {eps:.6} (scale-based bound)"
            );
        }
    }

    let global_max_eps = epsilon_per_channel.iter().copied().fold(0.0f32, f32::max);
    eprintln!(
        "Small INT8 epsilon proof: max analytical epsilon={global_max_eps:.6}, \
         all {SMALL_SEQ}x{SMALL_OUT} outputs verified within bound"
    );
}

/// Prove medium layer epsilon-bounded deviation under INT8 quantization.
#[test]
fn test_int8_medium_epsilon_deviation_proof() {
    let weights_f32 = xavier_weights(MED_OUT, MED_IN);
    let weights_int8 = quantize_int8_symmetric(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[MED_OUT]), 0.0f32);

    // Compute analytical epsilon per output channel.
    let shape = weights_f32.shape();
    let (out_ch, in_ch) = (shape[0], shape[1]);
    let mut epsilon_per_channel = Vec::with_capacity(out_ch);
    for oc in 0..out_ch {
        let row_max = (0..in_ch)
            .map(|ic| weights_f32[[oc, ic]].abs())
            .fold(0.0f32, f32::max);
        if row_max == 0.0 {
            epsilon_per_channel.push(0.0);
            continue;
        }
        let scale = row_max / 127.0;
        let eps = (scale / 2.0) * in_ch as f32 * VIT_RANGE;
        epsilon_per_channel.push(eps);
    }

    let def_f32 = build_linear_kernel("int8_med_eps_f32", MED_SEQ, MED_IN, MED_OUT);
    let def_q = build_linear_kernel("int8_med_eps_q", MED_SEQ, MED_IN, MED_OUT);

    let bindings_f32 = linear_bindings(&weights_f32, &bias);
    let bindings_q = linear_bindings(&weights_int8, &bias);

    let graph_f32 = tensor_kernel_to_graph(&def_f32, &bindings_f32).expect("f32 graph");
    let graph_q = tensor_kernel_to_graph(&def_q, &bindings_q).expect("int8 graph");

    let input = uniform_bounds(&[MED_SEQ, MED_IN], VIT_RANGE);

    let out_f32 = graph_f32.propagate_ibp(&input).expect("f32 IBP");
    let out_q = graph_q.propagate_ibp(&input).expect("int8 IBP");

    let (f32_lo, f32_hi) = out_f32.lower_upper();
    let (q_lo, q_hi) = out_q.lower_upper();

    for s in 0..MED_SEQ {
        for oc in 0..MED_OUT {
            let lo_drift = (f32_lo[[s, oc]] - q_lo[[s, oc]]).abs();
            let hi_drift = (f32_hi[[s, oc]] - q_hi[[s, oc]]).abs();
            let max_ch_drift = lo_drift.max(hi_drift);
            let eps = epsilon_per_channel[oc];

            assert!(
                max_ch_drift <= eps + 1e-3,
                "seq={s}, ch={oc}: drift {max_ch_drift:.6} exceeds analytical \
                 epsilon {eps:.6} (scale-based bound)"
            );
        }
    }

    let global_max_eps = epsilon_per_channel.iter().copied().fold(0.0f32, f32::max);
    eprintln!(
        "Medium INT8 epsilon proof: max analytical epsilon={global_max_eps:.6}, \
         all {MED_SEQ}x{MED_OUT} outputs verified within bound"
    );
}

/// Prove asymmetric zero_point perturbation epsilon bound for small layer.
///
/// For asymmetric quantization, the per-element error bound is scale/2
/// (same as symmetric), but the effective perturbation also includes
/// the zero_point offset. This test verifies per-channel epsilon holds.
#[test]
fn test_int8_small_asymmetric_epsilon_deviation_proof() {
    let weights_f32 = xavier_weights(SMALL_OUT, SMALL_IN);
    let weights_asym = quantize_int8_asymmetric(&weights_f32);
    let (scales, _zero_points) = asymmetric_scales_and_zero_points(&weights_f32);
    let bias = ArrayD::from_elem(IxDyn(&[SMALL_OUT]), 0.0f32);

    let shape = weights_f32.shape();
    let in_ch = shape[1];

    // Analytical epsilon for asymmetric: (scale/2) * in_features * VIT_RANGE.
    let epsilon_per_channel: Vec<f32> = scales
        .iter()
        .map(|&scale| {
            if scale == 0.0 {
                0.0
            } else {
                (scale / 2.0) * in_ch as f32 * VIT_RANGE
            }
        })
        .collect();

    let def_f32 = build_linear_kernel("int8_asym_eps_f32", SMALL_SEQ, SMALL_IN, SMALL_OUT);
    let def_q = build_linear_kernel("int8_asym_eps_q", SMALL_SEQ, SMALL_IN, SMALL_OUT);

    let bindings_f32 = linear_bindings(&weights_f32, &bias);
    let bindings_q = linear_bindings(&weights_asym, &bias);

    let graph_f32 = tensor_kernel_to_graph(&def_f32, &bindings_f32).expect("f32 graph");
    let graph_q = tensor_kernel_to_graph(&def_q, &bindings_q).expect("int8 graph");

    let input = uniform_bounds(&[SMALL_SEQ, SMALL_IN], VIT_RANGE);

    let out_f32 = graph_f32.propagate_ibp(&input).expect("f32 IBP");
    let out_q = graph_q.propagate_ibp(&input).expect("asymmetric int8 IBP");

    let (f32_lo, f32_hi) = out_f32.lower_upper();
    let (q_lo, q_hi) = out_q.lower_upper();

    for s in 0..SMALL_SEQ {
        for oc in 0..SMALL_OUT {
            let lo_drift = (f32_lo[[s, oc]] - q_lo[[s, oc]]).abs();
            let hi_drift = (f32_hi[[s, oc]] - q_hi[[s, oc]]).abs();
            let max_ch_drift = lo_drift.max(hi_drift);
            let eps = epsilon_per_channel[oc];

            assert!(
                max_ch_drift <= eps + 1e-4,
                "Asymmetric seq={s}, ch={oc}: drift {max_ch_drift:.6} exceeds \
                 analytical epsilon {eps:.6}"
            );
        }
    }

    let global_max_eps = epsilon_per_channel.iter().copied().fold(0.0f32, f32::max);
    eprintln!(
        "Small asymmetric INT8 epsilon proof: max epsilon={global_max_eps:.6}, \
         all outputs verified within bound"
    );
}
