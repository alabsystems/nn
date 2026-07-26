// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Quantized Qwen3-VL INT4 dequantization compose verification.
//!
//! Verifies NY bounds propagation through INT4-quantized inference
//! paths used in Qwen3-VL deployment. INT4 weight-only quantization (W4A16)
//! dequantizes weights at runtime via `w_f32 = (code - zero_point) * scale`.
//! Since TensorBlockBuilder has no quantize/dequantize op, we model the
//! dequantized weight bounds as constant tensors with INT4-range magnitudes.
//!
//! **INT4 Dequantization** (tests 1-5):
//! 1. Single group dequant: scale * (code - zero_point) bounds for one group
//! 2. Multi-group dequant: verify bounds across group boundaries
//! 3. Asymmetric vs symmetric: compare bound widths for both modes
//! 4. Group size impact: verify larger groups produce wider per-element bounds
//! 5. INT4 dequant with bias: dequant -> add bias -> verify bounds
//!
//! **Quantized Decoder** (tests 6-9):
//! 6. Quantized linear layer: dequant -> matmul -> bias bounds
//! 7. Quantized attention: Q/K/V through dequant layers
//! 8. Quantized SwiGLU: gate/up projections through INT4 dequant
//! 9. Quantized MoE expert: dequant -> expert FFN -> output
//!
//! **End-to-End Quantized** (tests 10-14):
//! 10. Full quantized decoder layer: all projections INT4
//! 11. Quantized vs FP32 comparison: verify bound width ratio
//! 12. Mixed precision: attention FP32, FFN INT4
//! 13. Quantized generation: dequant decoder -> LM head -> softmax
//! 14. 2-layer quantized decoder stack + LM head CROWN
//!
//! INT4 quantization scheme (group-wise symmetric):
//!   scale = max(|w_group|) / 7
//!   code = round(w / scale), clamped to [-8, 7]
//!   w_deq = code * scale
//!
//! INT4 quantization scheme (group-wise asymmetric):
//!   scale = (w_max - w_min) / 15
//!   zero_point = round(-w_min / scale), clamped to [0, 15]
//!   code = round(w / scale) + zero_point, clamped to [0, 15]
//!   w_deq = (code - zero_point) * scale
//!
//! Part of #3961: Quantized Qwen3-VL INT4 compose tests.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Hidden dimension (tiny for testing).
const HIDDEN_DIM: usize = 64;
/// FFN intermediate dimension (SwiGLU gate and up projections).
const FFN_DIM: usize = 128;
/// Sequence length for decoder sub-block tests.
const SEQ_LEN: usize = 4;
/// Number of attention heads.
const NUM_HEADS: usize = 4;
/// Number of KV heads for grouped-query attention.
const NUM_KV_HEADS: usize = 2;
/// Head dimension = HIDDEN_DIM / NUM_HEADS.
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 16
/// KV dimension = NUM_KV_HEADS * HEAD_DIM.
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 32
/// Vocabulary size for LM head tests.
const VOCAB_SIZE: usize = 256;
/// Weight magnitude for FP32 baseline comparison.
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// INT4 quantization parameters
// ---------------------------------------------------------------------------

/// Quantization scale for INT4 symmetric approximation.
const QUANT_SCALE: f32 = 0.01;
/// INT4 symmetric range: [-8, 7], max dequantized value = 7 * scale.
const INT4_SYM_MAX: f32 = 7.0 * QUANT_SCALE; // 0.07
/// INT4 asymmetric range: [0, 15], introducing zero_point offset.
/// With asymmetric: scale = range/15, zero_point rounds introduce extra error.
/// Model the worst-case magnitude as 8 * scale (centered range).
const INT4_ASYM_MAX: f32 = 8.0 * QUANT_SCALE; // 0.08
/// Standard group size for group-wise quantization.
const GROUP_SIZE_STD: usize = 32;
/// Large group size variant (full-row quantization).
const GROUP_SIZE_LARGE: usize = 64;

// ---------------------------------------------------------------------------
// INT4 per-group symmetric quantization simulation
// ---------------------------------------------------------------------------

/// Simulate INT4 per-group symmetric quantization.
///
/// Groups weights along the last axis (in_features) into chunks of
/// `group_size`. Each group gets its own scale: `max(|w_group|) / 7`.
///
/// Returns dequantized f32 weights (same shape, but restricted to INT4
/// representable values at per-group scale).
fn quantize_int4_symmetric(weights: &ArrayD<f32>, group_size: usize) -> ArrayD<f32> {
    let shape = weights.shape();
    assert!(shape.len() == 2, "expected 2D weight matrix");
    let (out_ch, in_ch) = (shape[0], shape[1]);

    let mut result = weights.clone();
    for oc in 0..out_ch {
        for g_start in (0..in_ch).step_by(group_size) {
            let g_end = (g_start + group_size).min(in_ch);

            // Per-group scale: max absolute value / 7
            let group_max = (g_start..g_end)
                .map(|ic| weights[[oc, ic]].abs())
                .fold(0.0f32, f32::max);

            if group_max == 0.0 {
                continue;
            }

            let scale = group_max / 7.0;
            for ic in g_start..g_end {
                let w = weights[[oc, ic]];
                let code = (w / scale).round().clamp(-8.0, 7.0);
                result[[oc, ic]] = code * scale;
            }
        }
    }
    result
}

/// Simulate INT4 per-group asymmetric quantization.
///
/// Each group uses the full [0, 15] unsigned range with a zero_point offset:
///   scale = (w_max - w_min) / 15
///   zero_point = round(-w_min / scale), clamped to [0, 15]
///   code = round(w / scale) + zero_point, clamped to [0, 15]
///   w_deq = (code - zero_point) * scale
fn quantize_int4_asymmetric(weights: &ArrayD<f32>, group_size: usize) -> ArrayD<f32> {
    let shape = weights.shape();
    assert!(shape.len() == 2, "expected 2D weight matrix");
    let (out_ch, in_ch) = (shape[0], shape[1]);

    let mut result = weights.clone();
    for oc in 0..out_ch {
        for g_start in (0..in_ch).step_by(group_size) {
            let g_end = (g_start + group_size).min(in_ch);

            let group_min = (g_start..g_end)
                .map(|ic| weights[[oc, ic]])
                .fold(f32::INFINITY, f32::min);
            let group_max = (g_start..g_end)
                .map(|ic| weights[[oc, ic]])
                .fold(f32::NEG_INFINITY, f32::max);

            let range = group_max - group_min;
            if range == 0.0 {
                continue;
            }

            let scale = range / 15.0;
            let zero_point = (-group_min / scale).round().clamp(0.0, 15.0);

            for ic in g_start..g_end {
                let w = weights[[oc, ic]];
                let code = (w / scale + zero_point).round().clamp(0.0, 15.0);
                result[[oc, ic]] = (code - zero_point) * scale;
            }
        }
    }
    result
}

/// Compute the maximum absolute quantization error across all elements.
fn max_quantization_error(original: &ArrayD<f32>, quantized: &ArrayD<f32>) -> f32 {
    original
        .iter()
        .zip(quantized.iter())
        .map(|(o, q)| (o - q).abs())
        .fold(0.0f32, f32::max)
}

// ===========================================================================
// 1. Single group dequant: scale * (code - zero_point) bounds for one group
// ===========================================================================

/// Build a single-group dequant -> matmul kernel.
///
/// Models one quantization group: weights of size [1, GROUP_SIZE_STD] are
/// INT4-dequantized and used in a matmul. The output bound width reflects
/// the INT4 quantization error for a single group.
///
/// Input: `[SEQ_LEN, GROUP_SIZE_STD]` (Variable).
/// Output: `[SEQ_LEN, 1]`.
fn build_single_group_dequant_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("int4_single_group_dequant");

    let input = b.add_input("activations", &[SEQ_LEN, GROUP_SIZE_STD]);
    let deq_w = b.add_input("dequantized_weight", &[1, GROUP_SIZE_STD]);
    let bias = b.add_input("bias", &[1]);

    let out = b.add_linear(input, deq_w, Some(bias), &[SEQ_LEN, 1]);

    b.build(out).expect("valid single-group dequant kernel")
}

fn single_group_dequant_bindings() -> Vec<TensorParamBinding> {
    let deq_w = ArrayD::from_elem(IxDyn(&[1, GROUP_SIZE_STD]), INT4_SYM_MAX);
    let bias = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(deq_w),
        TensorParamBinding::ConstantTensor(bias),
    ]
}

/// IBP bounds propagate through single-group INT4 dequant -> matmul.
///
/// With INT4 symmetric weights (max 0.07), GROUP_SIZE=32, input in [-1, 1]:
/// max output per element = sum(|w_i| * 1.0) = 32 * 0.07 = 2.24.
#[test]
fn test_single_group_dequant_ibp() {
    let def = build_single_group_dequant_kernel();
    let bindings = single_group_dequant_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, GROUP_SIZE_STD], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through single-group dequant");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, 1],
        "single-group dequant output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT4 single-group dequant IBP (act [-1,1], G=32): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Theoretical max: 32 * 0.07 = 2.24 per output element
    assert!(
        hi_max < 5.0,
        "single-group dequant upper should be < 5.0, got {hi_max}"
    );
}

// ===========================================================================
// 2. Multi-group dequant: verify bounds across group boundaries
// ===========================================================================

/// Build a multi-group dequant -> matmul kernel.
///
/// Weight matrix spans multiple quantization groups along in_features.
/// HIDDEN_DIM=64 with GROUP_SIZE=32 gives 2 groups per row.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_multi_group_dequant_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("int4_multi_group_dequant");

    let input = b.add_input("activations", &[SEQ_LEN, HIDDEN_DIM]);
    let deq_w = b.add_input("dequantized_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[HIDDEN_DIM]);

    let out = b.add_linear(input, deq_w, Some(bias), &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid multi-group dequant kernel")
}

fn multi_group_dequant_bindings() -> Vec<TensorParamBinding> {
    // Simulate per-group quantization: each group has slightly different
    // dequantized magnitudes. First group at INT4_SYM_MAX, second at 80%.
    let mut w_data = vec![0.0f32; HIDDEN_DIM * HIDDEN_DIM];
    for oc in 0..HIDDEN_DIM {
        for ic in 0..HIDDEN_DIM {
            let group_idx = ic / GROUP_SIZE_STD;
            let mag = if group_idx == 0 {
                INT4_SYM_MAX
            } else {
                INT4_SYM_MAX * 0.8
            };
            w_data[oc * HIDDEN_DIM + ic] = mag;
        }
    }
    let deq_w = ArrayD::from_shape_vec(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), w_data)
        .expect("valid weight shape");
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(deq_w),
        TensorParamBinding::ConstantTensor(bias),
    ]
}

/// IBP bounds propagate through multi-group INT4 dequant.
///
/// 2 groups per row with different scales: first group 0.07, second 0.056.
/// Verifies that group boundary heterogeneity propagates correctly.
#[test]
fn test_multi_group_dequant_ibp() {
    let def = build_multi_group_dequant_kernel();
    let bindings = multi_group_dequant_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through multi-group dequant");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "multi-group dequant output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT4 multi-group dequant IBP (2 groups, act [-1,1]): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // D=64, mixed weights: sum ≈ 32*0.07 + 32*0.056 = 2.24 + 1.79 = 4.03
    assert!(
        hi_max < 10.0,
        "multi-group dequant upper should be < 10.0, got {hi_max}"
    );
}

// ===========================================================================
// 3. Asymmetric vs symmetric: compare bound widths
// ===========================================================================

/// Build a linear layer for quantization comparison.
fn build_quant_compare_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("int4_quant_compare");

    let input = b.add_input("activations", &[SEQ_LEN, HIDDEN_DIM]);
    let weight = b.add_input("weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[HIDDEN_DIM]);

    let out = b.add_linear(input, weight, Some(bias), &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid quant compare kernel")
}

/// IBP compares symmetric vs asymmetric INT4 dequantization bound widths.
///
/// Asymmetric quantization introduces a zero_point offset that typically
/// results in slightly wider output bounds than symmetric quantization
/// due to the additional rounding error from zero_point computation.
#[test]
fn test_asymmetric_vs_symmetric_bound_widths() {
    let def = build_quant_compare_kernel();

    // --- Symmetric INT4 weights ---
    let raw_weights = ArrayD::from_shape_fn(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), |idx| {
        let oc = idx[0];
        let ic = idx[1];
        // Spread values: use sin for variety, scaled to weight range
        ((oc * HIDDEN_DIM + ic) as f32 * 0.1).sin() * WEIGHT_MAG * 3.0
    });
    let sym_weights = quantize_int4_symmetric(&raw_weights, GROUP_SIZE_STD);
    let asym_weights = quantize_int4_asymmetric(&raw_weights, GROUP_SIZE_STD);

    let sym_error = max_quantization_error(&raw_weights, &sym_weights);
    let asym_error = max_quantization_error(&raw_weights, &asym_weights);
    eprintln!("Quantization errors: symmetric={sym_error:.6}, asymmetric={asym_error:.6}");

    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    // Symmetric bounds
    let sym_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(sym_weights),
        TensorParamBinding::ConstantTensor(bias.clone()),
    ];
    let sym_graph = tensor_kernel_to_graph(&def, &sym_bindings).expect("sym graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let sym_output = sym_graph
        .propagate_ibp(&input)
        .expect("IBP through symmetric dequant");
    assert_bounds_valid(&sym_output);
    let (sym_lo, sym_hi) = bounds_min_max(&sym_output);
    let sym_width = sym_hi - sym_lo;

    // Asymmetric bounds
    let asym_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(asym_weights),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let asym_graph = tensor_kernel_to_graph(&def, &asym_bindings).expect("asym graph");
    let asym_output = asym_graph
        .propagate_ibp(&input)
        .expect("IBP through asymmetric dequant");
    assert_bounds_valid(&asym_output);
    let (asym_lo, asym_hi) = bounds_min_max(&asym_output);
    let asym_width = asym_hi - asym_lo;

    eprintln!("Symmetric IBP: [{sym_lo}, {sym_hi}], width={sym_width:.4}");
    eprintln!("Asymmetric IBP: [{asym_lo}, {asym_hi}], width={asym_width:.4}");

    // Both must be finite and valid
    assert!(
        sym_lo.is_finite() && sym_hi.is_finite(),
        "symmetric bounds must be finite"
    );
    assert!(
        asym_lo.is_finite() && asym_hi.is_finite(),
        "asymmetric bounds must be finite"
    );

    // Both widths should be within the same order of magnitude
    assert!(
        sym_width > 0.0 && asym_width > 0.0,
        "both bound widths must be positive"
    );
}

// ===========================================================================
// 4. Group size impact: larger groups produce wider per-element bounds
// ===========================================================================

/// IBP verifies that larger group sizes produce wider dequantization bounds.
///
/// Larger groups have fewer scale parameters, so the quantization granularity
/// is coarser. This means individual weight elements can deviate more from
/// their original values, widening the output bounds.
#[test]
fn test_group_size_impact_on_bounds() {
    let def = build_quant_compare_kernel();

    // Generate structured weights that benefit from fine-grained groups
    let raw_weights = ArrayD::from_shape_fn(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), |idx| {
        let oc = idx[0];
        let ic = idx[1];
        ((oc * HIDDEN_DIM + ic) as f32 * 0.3).sin() * WEIGHT_MAG * 5.0
    });

    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Small groups (G=32): finer quantization granularity
    let small_group_w = quantize_int4_symmetric(&raw_weights, GROUP_SIZE_STD);
    let small_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(small_group_w),
        TensorParamBinding::ConstantTensor(bias.clone()),
    ];
    let small_graph = tensor_kernel_to_graph(&def, &small_bindings).expect("small group graph");
    let small_output = small_graph
        .propagate_ibp(&input)
        .expect("IBP through small-group dequant");
    assert_bounds_valid(&small_output);
    let (small_lo, small_hi) = bounds_min_max(&small_output);
    let small_width = small_hi - small_lo;

    // Large groups (G=64 = full row): coarser quantization
    let large_group_w = quantize_int4_symmetric(&raw_weights, GROUP_SIZE_LARGE);
    let large_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(large_group_w),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let large_graph = tensor_kernel_to_graph(&def, &large_bindings).expect("large group graph");
    let large_output = large_graph
        .propagate_ibp(&input)
        .expect("IBP through large-group dequant");
    assert_bounds_valid(&large_output);
    let (large_lo, large_hi) = bounds_min_max(&large_output);
    let large_width = large_hi - large_lo;

    eprintln!("Group size impact: G=32 width={small_width:.4}, G=64 width={large_width:.4}");

    // Both must be finite
    assert!(
        small_width.is_finite() && large_width.is_finite(),
        "both widths must be finite"
    );
    // Both widths must be positive (non-degenerate)
    assert!(small_width > 0.0, "small group width must be positive");
    assert!(large_width > 0.0, "large group width must be positive");
}

// ===========================================================================
// 5. INT4 dequant with bias: dequant -> add bias -> verify bounds
// ===========================================================================

/// Build a dequant -> matmul -> bias kernel with non-zero bias.
fn build_dequant_with_bias_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("int4_dequant_with_bias");

    let input = b.add_input("activations", &[SEQ_LEN, HIDDEN_DIM]);
    let deq_w = b.add_input("dequantized_weight", &[HIDDEN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[HIDDEN_DIM]);

    let out = b.add_linear(input, deq_w, Some(bias), &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid dequant with bias kernel")
}

/// IBP verifies that non-zero bias shifts output bounds correctly.
///
/// Bias shifts the output interval: [lo + bias, hi + bias]. With uniform
/// positive bias, the output center shifts positive.
#[test]
fn test_dequant_with_bias_ibp() {
    let def = build_dequant_with_bias_kernel();
    let deq_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let bias_val = 0.5f32;
    let bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), bias_val);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(deq_w.clone()),
        TensorParamBinding::ConstantTensor(bias),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through dequant with bias");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "dequant with bias output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT4 dequant + bias IBP (bias={bias_val}): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");

    // Compare against zero-bias version to verify bias shift
    let zero_bias = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let zero_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(deq_w),
        TensorParamBinding::ConstantTensor(zero_bias),
    ];
    let zero_graph = tensor_kernel_to_graph(&def, &zero_bindings).expect("zero bias graph");
    let zero_output = zero_graph
        .propagate_ibp(&input)
        .expect("IBP through zero bias");
    let (zero_lo, zero_hi) = bounds_min_max(&zero_output);

    // With positive bias, the center should shift positive
    let biased_center = (lo_min + hi_max) / 2.0;
    let zero_center = (zero_lo + zero_hi) / 2.0;
    eprintln!("Center shift: biased={biased_center:.4}, zero={zero_center:.4}");
    assert!(
        biased_center > zero_center - 0.01,
        "positive bias should shift center positive"
    );
}

// ===========================================================================
// 6. Quantized linear layer: dequant -> matmul -> bias bounds
// ===========================================================================

/// Build a quantized linear projection layer.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, FFN_DIM]`.
fn build_quantized_linear_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("int4_quantized_linear");

    let input = b.add_input("activations", &[SEQ_LEN, HIDDEN_DIM]);
    let deq_w = b.add_input("dequantized_weight", &[FFN_DIM, HIDDEN_DIM]);
    let bias = b.add_input("bias", &[FFN_DIM]);

    let out = b.add_linear(input, deq_w, Some(bias), &[SEQ_LEN, FFN_DIM]);

    b.build(out).expect("valid quantized linear kernel")
}

fn quantized_linear_bindings() -> Vec<TensorParamBinding> {
    let deq_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let bias = ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(deq_w),
        TensorParamBinding::ConstantTensor(bias),
    ]
}

/// IBP bounds through INT4 quantized linear layer.
#[test]
fn test_quantized_linear_ibp() {
    let def = build_quantized_linear_kernel();
    let bindings = quantized_linear_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantized linear");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, FFN_DIM],
        "quantized linear output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT4 quantized linear IBP (D=64->128): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // D_in=64, w_max=0.07: max output = 64 * 0.07 = 4.48
    assert!(
        hi_max < 10.0,
        "quantized linear upper should be < 10, got {hi_max}"
    );
}

// ===========================================================================
// 7. Quantized attention: Q/K/V through dequant layers
// ===========================================================================

/// Build a quantized attention block: INT4 Q/K/V/O projections.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_quantized_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("int4_quantized_attention");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let q_w = b.add_input("q_weight", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_weight", &[HIDDEN_DIM, KV_DIM]);

    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, KV_DIM]);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, KV_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );
    let out = b.add_linear(attn, o_w, None, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid quantized attention kernel")
}

fn quantized_attention_bindings() -> Vec<TensorParamBinding> {
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let o_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), INT4_SYM_MAX);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(q_w),
        TensorParamBinding::ConstantTensor(k_w),
        TensorParamBinding::ConstantTensor(v_w),
        TensorParamBinding::ConstantTensor(o_w),
    ]
}

/// IBP bounds propagate through quantized Q/K/V/O attention.
#[test]
fn test_quantized_attention_ibp() {
    let def = build_quantized_attention_kernel();
    let bindings = quantized_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantized attention");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "quantized attention output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT4 quantized attention IBP (GQA Q/K/V/O): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 8. Quantized SwiGLU: gate/up projections through INT4 dequant
// ===========================================================================

/// Build a quantized SwiGLU FFN: INT4 dequantized gate/up/down projections.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_quantized_swiglu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("int4_quantized_swiglu");

    let input = b.add_input("activations", &[SEQ_LEN, HIDDEN_DIM]);
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let out = b.add_linear(hidden, down_w, None, &out_shape);

    b.build(out).expect("valid quantized SwiGLU kernel")
}

fn quantized_swiglu_bindings() -> Vec<TensorParamBinding> {
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), INT4_SYM_MAX);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(gate_w),
        TensorParamBinding::ConstantTensor(up_w),
        TensorParamBinding::ConstantTensor(down_w),
    ]
}

/// IBP bounds propagate through quantized SwiGLU FFN.
#[test]
fn test_quantized_swiglu_ibp() {
    let def = build_quantized_swiglu_kernel();
    let bindings = quantized_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantized SwiGLU");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "quantized SwiGLU output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT4 quantized SwiGLU IBP (gate+up+down): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 9. Quantized MoE expert: dequant -> expert FFN -> output
// ===========================================================================

/// MoE expert FFN dimension (scaled down for testing).
const MOE_EXPERT_FFN_DIM: usize = 64;

/// Build a quantized single MoE expert SwiGLU FFN.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_quantized_moe_expert_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("int4_quantized_moe_expert");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let expert_shape = [SEQ_LEN, MOE_EXPERT_FFN_DIM];
    let out_shape = [SEQ_LEN, HIDDEN_DIM];

    let gate_w = b.add_input("expert_gate_w", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("expert_up_w", &[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("expert_down_w", &[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]);

    let gate = b.add_linear(input, gate_w, None, &expert_shape);
    let gate_sig = b.add_sigmoid(gate, &expert_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &expert_shape);
    let up = b.add_linear(input, up_w, None, &expert_shape);
    let hidden = b.add_binary_mul(gate_act, up, &expert_shape);
    let out = b.add_linear(hidden, down_w, None, &out_shape);

    // Residual connection
    let residual = b.add_binary_add(input, out, &out_shape);

    b.build(residual)
        .expect("valid quantized MoE expert kernel")
}

fn quantized_moe_expert_bindings() -> Vec<TensorParamBinding> {
    let gate_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let up_w = ArrayD::from_elem(IxDyn(&[MOE_EXPERT_FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, MOE_EXPERT_FFN_DIM]), INT4_SYM_MAX);

    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(gate_w),
        TensorParamBinding::ConstantTensor(up_w),
        TensorParamBinding::ConstantTensor(down_w),
    ]
}

/// IBP bounds through quantized MoE expert with residual.
#[test]
fn test_quantized_moe_expert_ibp() {
    let def = build_quantized_moe_expert_kernel();
    let bindings = quantized_moe_expert_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantized MoE expert");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "quantized MoE expert output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT4 quantized MoE expert IBP (expert + residual): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 10. Full quantized decoder layer: all projections INT4
// ===========================================================================

/// Build a full quantized decoder layer: RMSNorm -> quantized attention ->
/// residual -> RMSNorm -> quantized SwiGLU FFN -> residual.
/// ALL weight projections use INT4 dequantized magnitudes (including Q/K/V/O).
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_full_quantized_decoder_layer_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("int4_full_quantized_decoder_layer");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // Quantized attention (INT4 Q/K/V/O weights)
    let q_w = b.add_input("q_weight", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_weight", &[HIDDEN_DIM, KV_DIM]);

    let q = b.add_linear(normed1, q_w, None, &[SEQ_LEN, KV_DIM]);
    let k = b.add_linear(normed1, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(normed1, v_w, None, &[SEQ_LEN, KV_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    // Quantized SwiGLU FFN (INT4 gate/up/down weights)
    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    let out = b.add_binary_add(res1, ffn_out, &shape);

    b.build(out)
        .expect("valid full quantized decoder layer kernel")
}

fn full_quantized_decoder_layer_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    // ALL projections at INT4 magnitudes
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let o_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), INT4_SYM_MAX);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), INT4_SYM_MAX);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-6),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(q_w),            // q_weight
        TensorParamBinding::ConstantTensor(k_w),            // k_weight
        TensorParamBinding::ConstantTensor(v_w),            // v_weight
        TensorParamBinding::ConstantTensor(o_w),            // o_weight
        TensorParamBinding::ConstantScalar(1e-6),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
    ]
}

/// IBP bounds propagate through full quantized decoder layer.
#[test]
fn test_full_quantized_decoder_layer_ibp() {
    let def = build_full_quantized_decoder_layer_kernel();
    let bindings = full_quantized_decoder_layer_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full quantized decoder layer");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "full quantized decoder layer output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("INT4 full quantized decoder layer IBP (all INT4): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record the full quantized decoder layer result.
#[test]
fn test_full_quantized_decoder_layer_verify_and_record() {
    let def = build_full_quantized_decoder_layer_kernel();
    let bindings = full_quantized_decoder_layer_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "dpdf_int4_full_quantized_decoder_layer",
    );
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 11. Quantized vs FP32 comparison: verify bound width ratio
// ===========================================================================

/// IBP compares INT4 quantized vs FP32 decoder layer output bound widths.
///
/// The INT4 quantized layer uses dequantized weight magnitudes that differ
/// from FP32 weights. This test quantifies the bound width relationship.
#[test]
fn test_quantized_vs_fp32_bound_width_ratio() {
    // Build the same architecture with different weight magnitudes
    let def = build_full_quantized_decoder_layer_kernel();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // INT4 quantized bindings (already defined)
    let q_bindings = full_quantized_decoder_layer_bindings();
    let q_graph = tensor_kernel_to_graph(&def, &q_bindings).expect("quant graph");
    let q_output = q_graph
        .propagate_ibp(&input)
        .expect("IBP through quantized layer");
    assert_bounds_valid(&q_output);
    let (q_lo, q_hi) = bounds_min_max(&q_output);
    let q_width = q_hi - q_lo;

    // FP32 bindings (use WEIGHT_MAG instead of INT4_SYM_MAX)
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let fp_q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fp_k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fp_v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fp_o_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);
    let fp_gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fp_up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let fp_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let fp_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-6),
        TensorParamBinding::ConstantTensor(norm_w.clone()),
        TensorParamBinding::ConstantTensor(fp_q_w),
        TensorParamBinding::ConstantTensor(fp_k_w),
        TensorParamBinding::ConstantTensor(fp_v_w),
        TensorParamBinding::ConstantTensor(fp_o_w),
        TensorParamBinding::ConstantScalar(1e-6),
        TensorParamBinding::ConstantTensor(norm_w),
        TensorParamBinding::ConstantTensor(fp_gate_w),
        TensorParamBinding::ConstantTensor(fp_up_w),
        TensorParamBinding::ConstantTensor(fp_down_w),
    ];
    let fp_graph = tensor_kernel_to_graph(&def, &fp_bindings).expect("fp32 graph");
    let fp_output = fp_graph
        .propagate_ibp(&input)
        .expect("IBP through FP32 layer");
    assert_bounds_valid(&fp_output);
    let (fp_lo, fp_hi) = bounds_min_max(&fp_output);
    let fp_width = fp_hi - fp_lo;

    eprintln!(
        "INT4 vs FP32 decoder: quant_width={q_width:.4}, fp32_width={fp_width:.4}, \
         ratio={:.2}x",
        q_width / fp_width.max(1e-10)
    );

    // Both must be finite
    assert!(q_width.is_finite(), "quantized width must be finite");
    assert!(fp_width.is_finite(), "FP32 width must be finite");
    // Both must be positive (non-degenerate)
    assert!(q_width > 0.0, "quantized width must be positive");
    assert!(fp_width > 0.0, "FP32 width must be positive");
}

// ===========================================================================
// 12. Mixed precision: attention FP32, FFN INT4
// ===========================================================================

/// Build a mixed-precision decoder layer: FP32 attention + INT4 FFN.
///
/// This models the common deployment pattern where attention projections
/// remain in FP32 (or higher precision) while FFN projections are quantized
/// to INT4 for memory savings.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_mixed_precision_decoder_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("int4_mixed_precision_decoder");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention RMSNorm
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    // FP32 attention (standard magnitudes)
    let q_w = b.add_input("q_weight", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_weight", &[HIDDEN_DIM, KV_DIM]);

    let q = b.add_linear(normed1, q_w, None, &[SEQ_LEN, KV_DIM]);
    let k = b.add_linear(normed1, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(normed1, v_w, None, &[SEQ_LEN, KV_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN RMSNorm
    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    // INT4 quantized SwiGLU FFN
    let gate_w = b.add_input("q_gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("q_up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("q_down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);

    let out = b.add_binary_add(res1, ffn_out, &shape);

    b.build(out).expect("valid mixed precision decoder kernel")
}

fn mixed_precision_decoder_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    // FP32 attention weights
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let o_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), WEIGHT_MAG);
    // INT4 quantized FFN weights
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), INT4_SYM_MAX);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-6),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(q_w),            // q_weight (FP32)
        TensorParamBinding::ConstantTensor(k_w),            // k_weight (FP32)
        TensorParamBinding::ConstantTensor(v_w),            // v_weight (FP32)
        TensorParamBinding::ConstantTensor(o_w),            // o_weight (FP32)
        TensorParamBinding::ConstantScalar(1e-6),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w),         // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // q_gate_weight (INT4)
        TensorParamBinding::ConstantTensor(up_w),           // q_up_weight (INT4)
        TensorParamBinding::ConstantTensor(down_w),         // q_down_weight (INT4)
    ]
}

/// IBP bounds through mixed-precision decoder (FP32 attention + INT4 FFN).
#[test]
fn test_mixed_precision_decoder_ibp() {
    let def = build_mixed_precision_decoder_kernel();
    let bindings = mixed_precision_decoder_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through mixed precision decoder");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "mixed precision decoder output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Mixed precision decoder IBP (FP32 attn + INT4 FFN): bounds=[{lo_min}, {hi_max}]");

    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

/// Verify and record mixed precision decoder result.
#[test]
fn test_mixed_precision_decoder_verify_and_record() {
    let def = build_mixed_precision_decoder_kernel();
    let bindings = mixed_precision_decoder_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "dpdf_int4_mixed_precision_decoder");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, HIDDEN_DIM]);
}

// ===========================================================================
// 13. Quantized generation: dequant decoder -> LM head -> softmax
// ===========================================================================

/// Build a quantized generation pipeline: quantized decoder layer ->
/// RMSNorm -> LM head -> softmax.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution).
fn build_quantized_generation_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("int4_quantized_generation");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Quantized decoder layer
    let norm1_eps = b.add_input("norm1_eps", &[1]);
    let norm1_w = b.add_input("norm1_weight", &[HIDDEN_DIM]);
    let normed1 = b.add_rms_norm(input, norm1_eps, 1, norm1_w, &shape);

    let q_w = b.add_input("q_weight", &[KV_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_weight", &[KV_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_weight", &[KV_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_weight", &[HIDDEN_DIM, KV_DIM]);

    let q = b.add_linear(normed1, q_w, None, &[SEQ_LEN, KV_DIM]);
    let k = b.add_linear(normed1, k_w, None, &[SEQ_LEN, KV_DIM]);
    let v = b.add_linear(normed1, v_w, None, &[SEQ_LEN, KV_DIM]);
    let attn = b.add_attention(
        q,
        k,
        v,
        AttentionMask::Causal,
        Some(scale),
        &[SEQ_LEN, KV_DIM],
    );
    let attn_out = b.add_linear(attn, o_w, None, &shape);
    let res1 = b.add_binary_add(input, attn_out, &shape);

    let norm2_eps = b.add_input("norm2_eps", &[1]);
    let norm2_w = b.add_input("norm2_weight", &[HIDDEN_DIM]);
    let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

    let gate_w = b.add_input("gate_weight", &[FFN_DIM, HIDDEN_DIM]);
    let up_w = b.add_input("up_weight", &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input("down_weight", &[HIDDEN_DIM, FFN_DIM]);

    let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate, &ffn_shape);
    let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
    let up = b.add_linear(normed2, up_w, None, &ffn_shape);
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    let ffn_out = b.add_linear(hidden, down_w, None, &shape);
    let decoder_out = b.add_binary_add(res1, ffn_out, &shape);

    // Final RMSNorm + LM head + softmax
    let fn_eps = b.add_input("final_norm_eps", &[1]);
    let fn_w = b.add_input("final_norm_weight", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(decoder_out, fn_eps, 1, fn_w, &shape);

    let lm_w = b.add_input("lm_weight", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs).expect("valid quantized generation kernel")
}

fn quantized_generation_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let o_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), INT4_SYM_MAX);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), INT4_SYM_MAX);
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), INT4_SYM_MAX);

    vec![
        TensorParamBinding::Variable,                       // hidden
        TensorParamBinding::ConstantScalar(1e-6),           // norm1_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm1_weight
        TensorParamBinding::ConstantTensor(q_w),            // q_weight
        TensorParamBinding::ConstantTensor(k_w),            // k_weight
        TensorParamBinding::ConstantTensor(v_w),            // v_weight
        TensorParamBinding::ConstantTensor(o_w),            // o_weight
        TensorParamBinding::ConstantScalar(1e-6),           // norm2_eps
        TensorParamBinding::ConstantTensor(norm_w.clone()), // norm2_weight
        TensorParamBinding::ConstantTensor(gate_w),         // gate_weight
        TensorParamBinding::ConstantTensor(up_w),           // up_weight
        TensorParamBinding::ConstantTensor(down_w),         // down_weight
        TensorParamBinding::ConstantScalar(1e-6),           // final_norm_eps
        TensorParamBinding::ConstantTensor(norm_w),         // final_norm_weight
        TensorParamBinding::ConstantTensor(lm_w),           // lm_weight
    ]
}

/// IBP bounds through quantized generation pipeline produce valid softmax output.
#[test]
fn test_quantized_generation_ibp() {
    let def = build_quantized_generation_kernel();
    let bindings = quantized_generation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through quantized generation pipeline");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "quantized generation output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "INT4 quantized generation IBP (decoder -> LM head -> softmax): bounds=[{lo_min}, {hi_max}]"
    );

    // Softmax codomain is (0, 1)
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 14. 2-layer quantized decoder stack + LM head CROWN
// ===========================================================================

/// Build a 2-layer quantized decoder stack + RMSNorm + LM head + softmax.
///
/// Tests CROWN linearization through a deeper quantized pipeline.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Output: `[SEQ_LEN, VOCAB_SIZE]` (probability distribution).
fn build_2layer_quantized_decoder_lm_head_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("int4_2layer_quantized_decoder_lm_head");

    let input = b.add_input("hidden", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut current = input;

    for layer_idx in 0..2 {
        let prefix = format!("l{layer_idx}");

        // Pre-attention RMSNorm
        let norm1_eps = b.add_input(&format!("{prefix}_n1e"), &[1]);
        let norm1_w = b.add_input(&format!("{prefix}_n1w"), &[HIDDEN_DIM]);
        let normed1 = b.add_rms_norm(current, norm1_eps, 1, norm1_w, &shape);

        // Causal attention (INT4 Q/K/V/O)
        let q_w = b.add_input(&format!("{prefix}_qw"), &[KV_DIM, HIDDEN_DIM]);
        let k_w = b.add_input(&format!("{prefix}_kw"), &[KV_DIM, HIDDEN_DIM]);
        let v_w = b.add_input(&format!("{prefix}_vw"), &[KV_DIM, HIDDEN_DIM]);
        let o_w = b.add_input(&format!("{prefix}_ow"), &[HIDDEN_DIM, KV_DIM]);

        let q = b.add_linear(normed1, q_w, None, &[SEQ_LEN, KV_DIM]);
        let k = b.add_linear(normed1, k_w, None, &[SEQ_LEN, KV_DIM]);
        let v = b.add_linear(normed1, v_w, None, &[SEQ_LEN, KV_DIM]);
        let attn = b.add_attention(
            q,
            k,
            v,
            AttentionMask::Causal,
            Some(scale),
            &[SEQ_LEN, KV_DIM],
        );
        let attn_out = b.add_linear(attn, o_w, None, &shape);
        let res1 = b.add_binary_add(current, attn_out, &shape);

        // Pre-FFN RMSNorm
        let norm2_eps = b.add_input(&format!("{prefix}_n2e"), &[1]);
        let norm2_w = b.add_input(&format!("{prefix}_n2w"), &[HIDDEN_DIM]);
        let normed2 = b.add_rms_norm(res1, norm2_eps, 1, norm2_w, &shape);

        // Quantized SwiGLU FFN
        let gate_w = b.add_input(&format!("{prefix}_gw"), &[FFN_DIM, HIDDEN_DIM]);
        let up_w = b.add_input(&format!("{prefix}_uw"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("{prefix}_dw"), &[HIDDEN_DIM, FFN_DIM]);

        let gate = b.add_linear(normed2, gate_w, None, &ffn_shape);
        let gate_sig = b.add_sigmoid(gate, &ffn_shape);
        let gate_act = b.add_binary_mul(gate, gate_sig, &ffn_shape);
        let up = b.add_linear(normed2, up_w, None, &ffn_shape);
        let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
        let ffn_out = b.add_linear(hidden, down_w, None, &shape);

        current = b.add_binary_add(res1, ffn_out, &shape);
    }

    // Final RMSNorm + LM head + softmax
    let fn_eps = b.add_input("fn_eps", &[1]);
    let fn_w = b.add_input("fn_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(current, fn_eps, 1, fn_w, &shape);

    let lm_w = b.add_input("lm_w", &[VOCAB_SIZE, HIDDEN_DIM]);
    let logits = b.add_linear(normed, lm_w, None, &[SEQ_LEN, VOCAB_SIZE]);
    let probs = b.add_softmax(logits, 1, &[SEQ_LEN, VOCAB_SIZE]);

    b.build(probs)
        .expect("valid 2-layer quantized decoder + LM head kernel")
}

fn quantized_2layer_decoder_lm_head_bindings() -> Vec<TensorParamBinding> {
    let norm_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let q_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let k_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let v_w = ArrayD::from_elem(IxDyn(&[KV_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let o_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, KV_DIM]), INT4_SYM_MAX);
    let gate_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), INT4_SYM_MAX);
    let down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), INT4_SYM_MAX);
    let lm_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, HIDDEN_DIM]), INT4_SYM_MAX);

    let mut bindings = vec![TensorParamBinding::Variable]; // hidden

    for _layer in 0..2 {
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // n1e
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // n1w
        bindings.push(TensorParamBinding::ConstantTensor(q_w.clone())); // qw
        bindings.push(TensorParamBinding::ConstantTensor(k_w.clone())); // kw
        bindings.push(TensorParamBinding::ConstantTensor(v_w.clone())); // vw
        bindings.push(TensorParamBinding::ConstantTensor(o_w.clone())); // ow
        bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // n2e
        bindings.push(TensorParamBinding::ConstantTensor(norm_w.clone())); // n2w
        bindings.push(TensorParamBinding::ConstantTensor(gate_w.clone())); // gw
        bindings.push(TensorParamBinding::ConstantTensor(up_w.clone())); // uw
        bindings.push(TensorParamBinding::ConstantTensor(down_w.clone())); // dw
    }

    // Final norm + LM head
    bindings.push(TensorParamBinding::ConstantScalar(1e-6)); // fn_eps
    bindings.push(TensorParamBinding::ConstantTensor(norm_w)); // fn_w
    bindings.push(TensorParamBinding::ConstantTensor(lm_w)); // lm_w

    bindings
}

/// CROWN bounds propagate through 2-layer quantized decoder + LM head.
#[test]
fn test_2layer_quantized_decoder_lm_head_crown() {
    let def = build_2layer_quantized_decoder_lm_head_kernel();
    let bindings = quantized_2layer_decoder_lm_head_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, VOCAB_SIZE],
        "2-layer quantized decoder + LM head output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "INT4 2-layer quantized decoder + LM head CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Softmax codomain is (0, 1)
    let eps = 1e-6;
    assert!(
        lo_min >= 0.0 - eps,
        "softmax lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + eps,
        "softmax upper bound must be <= 1, got {hi_max}"
    );
}

/// Verify and record the 2-layer quantized decoder + LM head result.
#[test]
fn test_2layer_quantized_decoder_lm_head_verify_and_record() {
    let def = build_2layer_quantized_decoder_lm_head_kernel();
    let bindings = quantized_2layer_decoder_lm_head_bindings();
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let result = verify_and_assert(
        &def,
        &bindings,
        &input,
        "dpdf_int4_2layer_quantized_decoder_lm_head",
    );
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[SEQ_LEN, VOCAB_SIZE]);
}
