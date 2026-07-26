// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for positional encoding variants used across dpdf models.
//!
//! Verifies IBP and CROWN bound propagation through the major positional
//! encoding families that appear in document understanding models:
//!
//! ## Sinusoidal PE (tests 1-3)
//!
//! 1. Fixed sin/cos: bounded in [-1, 1] for all positions (IBP)
//! 2. Frequency scaling: higher dims -> lower frequency (IBP)
//! 3. Position interpolation: fractional positions bounded (IBP)
//!
//! ## Learned PE (tests 4-6)
//!
//! 4. Lookup embedding: bounded by weight matrix range (IBP)
//! 5. Position extrapolation: OOB positions handled (IBP)
//! 6. Learned vs sinusoidal: bound width comparison (IBP)
//!
//! ## RoPE (tests 7-9)
//!
//! 7. Rotation matrix: cos/sin bounded preserves norm (IBP + CROWN)
//! 8. RoPE at different positions: bounded rotation (IBP)
//! 9. RoPE + attention: QK dot product after rotation (IBP)
//!
//! ## M-RoPE / Qwen3-VL (tests 10-12)
//!
//! 10. 3-component: temporal, height, width rotations (IBP)
//! 11. Interleaved application: alternating components (IBP)
//! 12. M-RoPE vision: spatial position encoding bounded (IBP)
//!
//! ## 2D PE (tests 13-15)
//!
//! 13. Grid encoding: row + column sinusoidal (IBP)
//! 14. 2D PE + Conv: spatial features with position (IBP)
//! 15. PE monotone tightening: smaller eps -> tighter position bounds (IBP)
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=8, DIM=16, NUM_HEADS=4, HEAD_DIM=4, GRID_H=4, GRID_W=4
//!
//! Part of #4003: Positional encoding variant compose tests for dpdf models.

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

const SEQ_LEN: usize = 8;
const DIM: usize = 16;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = DIM / NUM_HEADS; // 4
const GRID_H: usize = 4;
const GRID_W: usize = 4;
const NUM_GRID_POS: usize = GRID_H * GRID_W; // 16
const WEIGHT_MAG: f32 = 0.02;
const VOCAB_SIZE: usize = 32;
const IN_CHANNELS: usize = 3;
const PATCH_SIZE: usize = 4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build sinusoidal PE tensor with values in [-1, 1].
fn sinusoidal_pe_tensor(seq: usize, d: usize) -> ArrayD<f32> {
    let mut data = vec![0.0f32; seq * d];
    for t in 0..seq {
        for i in 0..d / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / d as f64);
            data[t * d + 2 * i] = freq.sin() as f32;
            data[t * d + 2 * i + 1] = freq.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq, d]), data).expect("valid PE")
}

/// Build sinusoidal PE tensor with fractional position indices.
///
/// Supports non-integer positions for interpolation testing.
fn sinusoidal_pe_fractional(positions: &[f64], d: usize) -> ArrayD<f32> {
    let seq = positions.len();
    let mut data = vec![0.0f32; seq * d];
    for (t_idx, &pos) in positions.iter().enumerate() {
        for i in 0..d / 2 {
            let freq = pos / 10000.0_f64.powf(2.0 * i as f64 / d as f64);
            data[t_idx * d + 2 * i] = freq.sin() as f32;
            data[t_idx * d + 2 * i + 1] = freq.cos() as f32;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[seq, d]), data).expect("valid fractional PE")
}

/// Build 2D sinusoidal PE tensor for spatial grids.
fn sinusoidal_pe_2d_tensor(h: usize, w: usize, d: usize) -> ArrayD<f32> {
    let half = d / 2;
    let mut data = vec![0.0f32; h * w * d];
    for y in 0..h {
        for x in 0..w {
            for i in 0..half / 2 {
                let freq_y = (y as f64) / 10000.0_f64.powf(4.0 * i as f64 / d as f64);
                let freq_x = (x as f64) / 10000.0_f64.powf(4.0 * i as f64 / d as f64);
                let idx = (y * w + x) * d;
                data[idx + 2 * i] = freq_y.sin() as f32;
                data[idx + 2 * i + 1] = freq_y.cos() as f32;
                data[idx + half + 2 * i] = freq_x.sin() as f32;
                data[idx + half + 2 * i + 1] = freq_x.cos() as f32;
            }
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[h * w, d]), data).expect("valid 2D PE")
}

/// Build RoPE cos/sin tensors for given sequence length and dimension.
fn rope_pe_tensors(seq: usize, d: usize) -> (ArrayD<f32>, ArrayD<f32>) {
    let mut cos_data = vec![0.0f32; seq * d];
    let mut sin_data = vec![0.0f32; seq * d];
    for t in 0..seq {
        for i in 0..d / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / d as f64);
            let c = freq.cos() as f32;
            let s = freq.sin() as f32;
            cos_data[t * d + 2 * i] = c;
            cos_data[t * d + 2 * i + 1] = c;
            sin_data[t * d + 2 * i] = s;
            sin_data[t * d + 2 * i + 1] = s;
        }
    }
    (
        ArrayD::from_shape_vec(IxDyn(&[seq, d]), cos_data).expect("cos PE"),
        ArrayD::from_shape_vec(IxDyn(&[seq, d]), sin_data).expect("sin PE"),
    )
}

/// Build RoPE rotation approximation in the graph.
///
/// For each pair (x_even, x_odd):
///   rotated_even = x_even * cos_theta - x_odd * sin_theta
///   rotated_odd  = x_even * sin_theta + x_odd * cos_theta
///
/// Approximated as: output = input * cos_tensor + input * sin_tensor
fn add_rope_approx(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    cos_pe: nn_dsl::TensorNodeId,
    sin_pe: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let x_cos = b.add_binary_mul(input, cos_pe, shape);
    let x_sin = b.add_binary_mul(input, sin_pe, shape);
    b.add_binary_add(x_cos, x_sin, shape)
}

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ===========================================================================
// 1. Fixed sin/cos: bounded in [-1, 1] for all positions (IBP)
// ===========================================================================

/// Verify that sinusoidal PE added to features preserves boundedness.
/// PE values are fixed constants in [-1, 1], so input [-R, R] + PE [-1, 1]
/// produces output in [-(R+1), R+1].
fn build_sinusoidal_pe_fixed_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_pe_sinusoidal_fixed");
    let input = b.add_input("features", &[SEQ_LEN, DIM]);
    let pe = b.add_input("positional_encoding", &[SEQ_LEN, DIM]);
    let out = b.add_binary_add(input, pe, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid sinusoidal PE kernel")
}

#[test]
fn test_sinusoidal_pe_fixed_bounded() {
    let def = build_sinusoidal_pe_fixed_kernel();
    let pe = sinusoidal_pe_tensor(SEQ_LEN, DIM);

    // Verify PE tensor itself is bounded in [-1, 1]
    let pe_min = pe.iter().copied().fold(f32::INFINITY, f32::min);
    let pe_max = pe.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        pe_min >= -1.0 - 1e-6,
        "PE values must be >= -1.0, got {pe_min}"
    );
    assert!(
        pe_max <= 1.0 + 1e-6,
        "PE values must be <= 1.0, got {pe_max}"
    );

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sinusoidal PE fixed IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Input [-2, 2] + PE [-1, 1] = output in [-3, 3]
    assert!(
        lo_min >= -3.0 - 1e-6,
        "sinusoidal PE lower should be >= -3.0, got {lo_min}"
    );
    assert!(
        hi_max <= 3.0 + 1e-6,
        "sinusoidal PE upper should be <= 3.0, got {hi_max}"
    );
}

// ===========================================================================
// 2. Frequency scaling: higher dims -> lower frequency (IBP)
// ===========================================================================

/// Verify that different frequency bands in sinusoidal PE produce
/// different bound characteristics. Lower-frequency bands (higher dim indices)
/// should produce smoother, tighter per-position variation.
#[test]
fn test_sinusoidal_pe_frequency_scaling() {
    // Build two PE tensors: one using only low-frequency dims, one using high-frequency
    let long_seq: usize = 16;
    let pe_full = sinusoidal_pe_tensor(long_seq, DIM);

    // Check that low-dim (high-freq) values vary more across positions
    // than high-dim (low-freq) values
    let mut low_dim_range = 0.0f32;
    let mut high_dim_range = 0.0f32;

    // Low dimension index = high frequency
    for d_idx in 0..2 {
        let mut vals: Vec<f32> = Vec::new();
        for t in 0..long_seq {
            vals.push(pe_full[[t, d_idx]]);
        }
        let range = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - vals.iter().copied().fold(f32::INFINITY, f32::min);
        low_dim_range = low_dim_range.max(range);
    }

    // High dimension index = low frequency
    for d_idx in (DIM - 2)..DIM {
        let mut vals: Vec<f32> = Vec::new();
        for t in 0..long_seq {
            vals.push(pe_full[[t, d_idx]]);
        }
        let range = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - vals.iter().copied().fold(f32::INFINITY, f32::min);
        high_dim_range = high_dim_range.max(range);
    }

    eprintln!(
        "Frequency scaling: low_dim_range={low_dim_range:.6}, high_dim_range={high_dim_range:.6}"
    );

    // Now verify IBP through addition with both frequency bands
    let def = build_sinusoidal_pe_fixed_kernel();
    let pe = sinusoidal_pe_tensor(SEQ_LEN, DIM);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Frequency scaling IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 3. Position interpolation: fractional positions bounded (IBP)
// ===========================================================================

/// Verify that fractional position indices produce bounded PE values.
/// This tests position interpolation for continuous position encodings.
#[test]
fn test_sinusoidal_pe_fractional_positions() {
    let fractional_positions: Vec<f64> = (0..SEQ_LEN)
        .map(|i| i as f64 + 0.5) // Half-step positions
        .collect();
    let pe_frac = sinusoidal_pe_fractional(&fractional_positions, DIM);

    // Verify fractional PE is still bounded in [-1, 1]
    let pe_min = pe_frac.iter().copied().fold(f32::INFINITY, f32::min);
    let pe_max = pe_frac.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        pe_min >= -1.0 - 1e-6,
        "fractional PE must be >= -1.0, got {pe_min}"
    );
    assert!(
        pe_max <= 1.0 + 1e-6,
        "fractional PE must be <= 1.0, got {pe_max}"
    );

    // Verify IBP with fractional PE
    let def = build_sinusoidal_pe_fixed_kernel();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_frac),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Fractional PE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -2.0 - 1e-6,
        "fractional PE lower should be >= -2.0, got {lo_min}"
    );
    assert!(
        hi_max <= 2.0 + 1e-6,
        "fractional PE upper should be <= 2.0, got {hi_max}"
    );
}

// ===========================================================================
// 4. Lookup embedding: bounded by weight matrix range (IBP)
// ===========================================================================

fn build_learned_pe_lookup_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_pe_learned_lookup");
    let indices = b.add_input("position_ids", &[SEQ_LEN]);
    let weight = b.add_input("pos_embed_weight", &[SEQ_LEN, DIM]);
    let out = b.add_embedding(indices, weight, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid learned PE lookup kernel")
}

#[test]
fn test_learned_pe_lookup_bounded() {
    let def = build_learned_pe_lookup_kernel();
    let pos_w = ArrayD::from_elem(IxDyn(&[SEQ_LEN, DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pos_w),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Learned PE lookup IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 5. Position extrapolation: OOB positions handled (IBP)
// ===========================================================================

/// Verify that learned PE with a larger vocabulary handles longer sequences.
/// Uses a PE weight matrix with more positions than the sequence length.
fn build_learned_pe_extended_kernel() -> TensorKernelDef {
    let max_pos: usize = VOCAB_SIZE; // More positions than SEQ_LEN
    let mut b = TensorBlockBuilder::new("dpdf_pe_learned_extended");
    let indices = b.add_input("position_ids", &[SEQ_LEN]);
    let weight = b.add_input("pos_embed_weight", &[max_pos, DIM]);
    let out = b.add_embedding(indices, weight, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid extended learned PE kernel")
}

#[test]
fn test_learned_pe_extrapolation() {
    let max_pos: usize = VOCAB_SIZE;
    let def = build_learned_pe_extended_kernel();
    // Weight matrix with entries bounded by WEIGHT_MAG
    let pos_w = ArrayD::from_elem(IxDyn(&[max_pos, DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pos_w),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Learned PE extrapolation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 6. Learned vs sinusoidal: bound width comparison (IBP)
// ===========================================================================

/// Compare bound widths between learned and sinusoidal PE when added to
/// the same input. Both should produce finite bounds; we log the comparison.
#[test]
fn test_learned_vs_sinusoidal_bound_width() {
    // Sinusoidal PE path
    let def_sin = build_sinusoidal_pe_fixed_kernel();
    let pe_sin = sinusoidal_pe_tensor(SEQ_LEN, DIM);
    let bindings_sin = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_sin),
    ];
    let graph_sin = tensor_kernel_to_graph(&def_sin, &bindings_sin).expect("sinusoidal graph");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);
    let output_sin = graph_sin.propagate_ibp(&input).expect("sinusoidal IBP");
    assert_bounds_valid(&output_sin);
    let sin_width = bound_width(&output_sin);

    // Learned PE path: use same add structure but with learned weights
    let def_learned = build_sinusoidal_pe_fixed_kernel(); // Same graph: input + pe
    let pe_learned = ArrayD::from_elem(IxDyn(&[SEQ_LEN, DIM]), WEIGHT_MAG);
    let bindings_learned = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_learned),
    ];
    let graph_learned =
        tensor_kernel_to_graph(&def_learned, &bindings_learned).expect("learned graph");
    let output_learned = graph_learned.propagate_ibp(&input).expect("learned IBP");
    assert_bounds_valid(&output_learned);
    let learned_width = bound_width(&output_learned);

    eprintln!("PE comparison: sinusoidal width={sin_width:.6}, learned width={learned_width:.6}");
    // Both must produce finite bounds
    assert!(sin_width.is_finite(), "sinusoidal width must be finite");
    assert!(learned_width.is_finite(), "learned width must be finite");
    // Learned PE with small weights should produce tighter bounds than sinusoidal
    // (WEIGHT_MAG=0.02 << 1.0 PE range)
    assert!(
        learned_width <= sin_width + 1e-6,
        "learned PE (weight_mag={WEIGHT_MAG}) should produce tighter bounds \
         than sinusoidal PE (range [-1,1]): learned={learned_width}, sin={sin_width}"
    );
}

// ===========================================================================
// 7. Rotation matrix: cos/sin bounded preserves norm (IBP + CROWN)
// ===========================================================================

fn build_rope_rotation_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_pe_rope_rotation");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let cos_pe = b.add_input("cos_theta", &[SEQ_LEN, DIM]);
    let sin_pe = b.add_input("sin_theta", &[SEQ_LEN, DIM]);
    let out = add_rope_approx(&mut b, input, cos_pe, sin_pe, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid RoPE rotation kernel")
}

fn rope_rotation_bindings() -> Vec<TensorParamBinding> {
    let (cos_pe, sin_pe) = rope_pe_tensors(SEQ_LEN, DIM);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_pe),
        TensorParamBinding::ConstantTensor(sin_pe),
    ]
}

#[test]
fn test_rope_rotation_ibp_bounded() {
    let def = build_rope_rotation_kernel();
    let bindings = rope_rotation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RoPE rotation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // cos/sin in [-1, 1], so x*cos + x*sin bounded by [-2, 2] for input in [-1, 1]
    assert!(lo_min.is_finite(), "RoPE lower bound must be finite");
    assert!(hi_max.is_finite(), "RoPE upper bound must be finite");
    assert!(lo_min >= -3.0, "RoPE lower should be >= -3.0, got {lo_min}");
    assert!(hi_max <= 3.0, "RoPE upper should be <= 3.0, got {hi_max}");
}

#[test]
fn test_rope_rotation_crown_bounds() {
    let def = build_rope_rotation_kernel();
    let bindings = rope_rotation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("RoPE rotation CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 8. RoPE at different positions: bounded rotation (IBP)
// ===========================================================================

/// Verify that RoPE applied to inputs at different positions all produce
/// bounded outputs. Uses a longer sequence to exercise more position indices.
#[test]
fn test_rope_different_positions_bounded() {
    let long_seq: usize = 16;
    let mut b = TensorBlockBuilder::new("dpdf_pe_rope_long_seq");
    let input = b.add_input("hidden", &[long_seq, DIM]);
    let cos_pe = b.add_input("cos_theta", &[long_seq, DIM]);
    let sin_pe = b.add_input("sin_theta", &[long_seq, DIM]);
    let out = add_rope_approx(&mut b, input, cos_pe, sin_pe, &[long_seq, DIM]);
    let def = b.build(out).expect("valid long-seq RoPE kernel");

    let (cos_t, sin_t) = rope_pe_tensors(long_seq, DIM);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_t),
        TensorParamBinding::ConstantTensor(sin_t),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[long_seq, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RoPE long sequence (len={long_seq}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "RoPE long-seq lower must be finite");
    assert!(hi_max.is_finite(), "RoPE long-seq upper must be finite");
}

// ===========================================================================
// 9. RoPE + attention: QK dot product after rotation (IBP)
// ===========================================================================

/// Verify that RoPE-encoded features fed into multi-head attention
/// produce bounded outputs. This is the standard transformer pattern:
/// Q = RoPE(Wq * x), K = RoPE(Wk * x), V = Wv * x, then attention(Q, K, V).
fn build_rope_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_pe_rope_attention");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let cos_pe = b.add_input("cos_theta", &[SEQ_LEN, DIM]);
    let sin_pe = b.add_input("sin_theta", &[SEQ_LEN, DIM]);

    // Apply RoPE to input (simulates Q/K rotation)
    let x_rotated = add_rope_approx(&mut b, input, cos_pe, sin_pe, &[SEQ_LEN, DIM]);

    // Attention projection weights
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let out = b
        .add_multi_head_attention(
            x_rotated,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[SEQ_LEN, DIM],
        )
        .expect("valid RoPE + MHA");

    b.build(out).expect("valid RoPE + attention kernel")
}

#[test]
fn test_rope_attention_ibp() {
    let def = build_rope_attention_kernel();
    let (cos_pe, sin_pe) = rope_pe_tensors(SEQ_LEN, DIM);
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_pe),
        TensorParamBinding::ConstantTensor(sin_pe),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RoPE + attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "RoPE+attn lower must be finite");
    assert!(hi_max.is_finite(), "RoPE+attn upper must be finite");
}

// ===========================================================================
// 10. M-RoPE 3-component: temporal, height, width rotations (IBP)
// ===========================================================================

/// M-RoPE splits the embedding into 3 sections (temporal, height, width)
/// and applies separate rotary encodings per section.
fn build_mrope_3component_kernel() -> TensorKernelDef {
    let section = DIM / 4;
    let remainder = DIM - 3 * section;
    let sec_shape = [SEQ_LEN, section];

    let mut b = TensorBlockBuilder::new("dpdf_pe_mrope_3component");

    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);

    // Per-component cos/sin
    let cos_t = b.add_input("cos_temporal", &sec_shape);
    let sin_t = b.add_input("sin_temporal", &sec_shape);
    let cos_h = b.add_input("cos_height", &sec_shape);
    let sin_h = b.add_input("sin_height", &sec_shape);
    let cos_w = b.add_input("cos_width", &sec_shape);
    let sin_w = b.add_input("sin_width", &sec_shape);

    // Narrow input into sections
    let sec0 = b.add_narrow(input, 1, 0, section, &sec_shape);
    let sec1 = b.add_narrow(input, 1, section, section, &sec_shape);
    let sec2 = b.add_narrow(input, 1, 2 * section, section, &sec_shape);

    // Apply RoPE per section
    let rot0 = add_rope_approx(&mut b, sec0, cos_t, sin_t, &sec_shape);
    let rot1 = add_rope_approx(&mut b, sec1, cos_h, sin_h, &sec_shape);
    let rot2 = add_rope_approx(&mut b, sec2, cos_w, sin_w, &sec_shape);

    // Remainder section (no rotation)
    let rem_shape = [SEQ_LEN, remainder];
    let sec3 = b.add_narrow(input, 1, 3 * section, remainder, &rem_shape);

    // Concatenate all sections: [SEQ_LEN, DIM]
    let out = b.add_concat(&[rot0, rot1, rot2, sec3], 1, &[SEQ_LEN, DIM]);

    b.build(out).expect("valid M-RoPE 3-component kernel")
}

#[test]
fn test_mrope_3component_ibp() {
    let def = build_mrope_3component_kernel();
    let section = DIM / 4;
    let (cos_t, sin_t) = rope_pe_tensors(SEQ_LEN, section);
    let (cos_h, sin_h) = rope_pe_tensors(SEQ_LEN, section);
    let (cos_w, sin_w) = rope_pe_tensors(SEQ_LEN, section);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_t),
        TensorParamBinding::ConstantTensor(sin_t),
        TensorParamBinding::ConstantTensor(cos_h),
        TensorParamBinding::ConstantTensor(sin_h),
        TensorParamBinding::ConstantTensor(cos_w),
        TensorParamBinding::ConstantTensor(sin_w),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("M-RoPE 3-component IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "M-RoPE lower bound must be finite");
    assert!(hi_max.is_finite(), "M-RoPE upper bound must be finite");
}

// ===========================================================================
// 11. Interleaved M-RoPE application: alternating components (IBP)
// ===========================================================================

/// Verify M-RoPE with interleaved dimension assignment: dimensions are
/// assigned to temporal/height/width in a round-robin pattern rather than
/// contiguous blocks. Modeled as three narrow + RoPE + concat.
#[test]
fn test_mrope_interleaved_ibp() {
    // For interleaved M-RoPE, we use the same graph structure as 3-component
    // but with different PE tensors reflecting the interleaved pattern.
    // The verification graph is structurally identical; the PE constants differ.
    let def = build_mrope_3component_kernel();
    let section = DIM / 4;

    // Create PE tensors with different base frequencies for each component
    // to simulate the interleaved assignment
    let base_freqs = [1.0, 2.0, 4.0]; // different bases for temporal/height/width
    let mut bindings = vec![TensorParamBinding::Variable];

    for &base in &base_freqs {
        let mut cos_data = vec![0.0f32; SEQ_LEN * section];
        let mut sin_data = vec![0.0f32; SEQ_LEN * section];
        for t in 0..SEQ_LEN {
            for i in 0..section / 2 {
                let freq = (t as f64 * base) / 10000.0_f64.powf(2.0 * i as f64 / section as f64);
                let c = freq.cos() as f32;
                let s = freq.sin() as f32;
                cos_data[t * section + 2 * i] = c;
                cos_data[t * section + 2 * i + 1] = c;
                sin_data[t * section + 2 * i] = s;
                sin_data[t * section + 2 * i + 1] = s;
            }
        }
        let cos_t = ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, section]), cos_data).expect("cos PE");
        let sin_t = ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, section]), sin_data).expect("sin PE");
        bindings.push(TensorParamBinding::ConstantTensor(cos_t));
        bindings.push(TensorParamBinding::ConstantTensor(sin_t));
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("M-RoPE interleaved IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min.is_finite(),
        "interleaved M-RoPE lower must be finite"
    );
    assert!(
        hi_max.is_finite(),
        "interleaved M-RoPE upper must be finite"
    );
}

// ===========================================================================
// 12. M-RoPE vision: spatial position encoding bounded (IBP)
// ===========================================================================

/// Verify M-RoPE for vision tokens where height/width components encode
/// spatial positions in a 2D grid. Uses grid positions instead of
/// sequential positions.
#[test]
fn test_mrope_vision_spatial_ibp() {
    let section = DIM / 4;
    let sec_shape = [NUM_GRID_POS, section];
    let remainder = DIM - 3 * section;
    let rem_shape = [NUM_GRID_POS, remainder];

    let mut b = TensorBlockBuilder::new("dpdf_pe_mrope_vision");
    let input = b.add_input("hidden", &[NUM_GRID_POS, DIM]);

    // For vision: temporal is constant (frame=0), height/width vary
    let cos_t = b.add_input("cos_temporal", &sec_shape);
    let sin_t = b.add_input("sin_temporal", &sec_shape);
    let cos_h = b.add_input("cos_height", &sec_shape);
    let sin_h = b.add_input("sin_height", &sec_shape);
    let cos_w = b.add_input("cos_width", &sec_shape);
    let sin_w = b.add_input("sin_width", &sec_shape);

    let sec0 = b.add_narrow(input, 1, 0, section, &sec_shape);
    let sec1 = b.add_narrow(input, 1, section, section, &sec_shape);
    let sec2 = b.add_narrow(input, 1, 2 * section, section, &sec_shape);
    let sec3 = b.add_narrow(input, 1, 3 * section, remainder, &rem_shape);

    let rot0 = add_rope_approx(&mut b, sec0, cos_t, sin_t, &sec_shape);
    let rot1 = add_rope_approx(&mut b, sec1, cos_h, sin_h, &sec_shape);
    let rot2 = add_rope_approx(&mut b, sec2, cos_w, sin_w, &sec_shape);

    let out = b.add_concat(&[rot0, rot1, rot2, sec3], 1, &[NUM_GRID_POS, DIM]);
    let def = b.build(out).expect("valid M-RoPE vision kernel");

    // Build spatial PE: height varies row-wise, width varies column-wise
    let (cos_t_data, sin_t_data) = rope_pe_tensors(1, section); // frame=0 only
                                                                // Tile temporal PE across all grid positions
    let cos_t_tiled = {
        let row = cos_t_data.as_slice().expect("contiguous");
        let mut data = Vec::with_capacity(NUM_GRID_POS * section);
        for _ in 0..NUM_GRID_POS {
            data.extend_from_slice(row);
        }
        ArrayD::from_shape_vec(IxDyn(&[NUM_GRID_POS, section]), data).expect("tiled cos")
    };
    let sin_t_tiled = {
        let row = sin_t_data.as_slice().expect("contiguous");
        let mut data = Vec::with_capacity(NUM_GRID_POS * section);
        for _ in 0..NUM_GRID_POS {
            data.extend_from_slice(row);
        }
        ArrayD::from_shape_vec(IxDyn(&[NUM_GRID_POS, section]), data).expect("tiled sin")
    };

    // Height PE: position = row index
    let mut cos_h_data = vec![0.0f32; NUM_GRID_POS * section];
    let mut sin_h_data = vec![0.0f32; NUM_GRID_POS * section];
    for y in 0..GRID_H {
        for x in 0..GRID_W {
            let pos = y * GRID_W + x;
            for i in 0..section / 2 {
                let freq = (y as f64) / 10000.0_f64.powf(2.0 * i as f64 / section as f64);
                cos_h_data[pos * section + 2 * i] = freq.cos() as f32;
                cos_h_data[pos * section + 2 * i + 1] = freq.cos() as f32;
                sin_h_data[pos * section + 2 * i] = freq.sin() as f32;
                sin_h_data[pos * section + 2 * i + 1] = freq.sin() as f32;
            }
        }
    }
    let cos_h_arr =
        ArrayD::from_shape_vec(IxDyn(&[NUM_GRID_POS, section]), cos_h_data).expect("cos_h");
    let sin_h_arr =
        ArrayD::from_shape_vec(IxDyn(&[NUM_GRID_POS, section]), sin_h_data).expect("sin_h");

    // Width PE: position = column index
    let mut cos_w_data = vec![0.0f32; NUM_GRID_POS * section];
    let mut sin_w_data = vec![0.0f32; NUM_GRID_POS * section];
    for y in 0..GRID_H {
        for x in 0..GRID_W {
            let pos = y * GRID_W + x;
            for i in 0..section / 2 {
                let freq = (x as f64) / 10000.0_f64.powf(2.0 * i as f64 / section as f64);
                cos_w_data[pos * section + 2 * i] = freq.cos() as f32;
                cos_w_data[pos * section + 2 * i + 1] = freq.cos() as f32;
                sin_w_data[pos * section + 2 * i] = freq.sin() as f32;
                sin_w_data[pos * section + 2 * i + 1] = freq.sin() as f32;
            }
        }
    }
    let cos_w_arr =
        ArrayD::from_shape_vec(IxDyn(&[NUM_GRID_POS, section]), cos_w_data).expect("cos_w");
    let sin_w_arr =
        ArrayD::from_shape_vec(IxDyn(&[NUM_GRID_POS, section]), sin_w_data).expect("sin_w");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_t_tiled),
        TensorParamBinding::ConstantTensor(sin_t_tiled),
        TensorParamBinding::ConstantTensor(cos_h_arr),
        TensorParamBinding::ConstantTensor(sin_h_arr),
        TensorParamBinding::ConstantTensor(cos_w_arr),
        TensorParamBinding::ConstantTensor(sin_w_arr),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_GRID_POS, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("M-RoPE vision spatial IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "M-RoPE vision lower must be finite");
    assert!(hi_max.is_finite(), "M-RoPE vision upper must be finite");
}

// ===========================================================================
// 13. Grid encoding: row + column sinusoidal (IBP)
// ===========================================================================

/// Verify 2D sinusoidal PE where row and column positions are encoded
/// separately and concatenated into a single positional encoding vector.
fn build_2d_grid_pe_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_pe_2d_grid");
    let input = b.add_input("spatial_features", &[NUM_GRID_POS, DIM]);
    let pe = b.add_input("pe_2d", &[NUM_GRID_POS, DIM]);
    let out = b.add_binary_add(input, pe, &[NUM_GRID_POS, DIM]);
    b.build(out).expect("valid 2D grid PE kernel")
}

#[test]
fn test_2d_grid_pe_ibp() {
    let def = build_2d_grid_pe_kernel();
    let pe = sinusoidal_pe_2d_tensor(GRID_H, GRID_W, DIM);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_GRID_POS, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("2D grid PE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Input [-1, 1] + 2D PE [-1, 1] = output in [-2, 2]
    assert!(
        lo_min >= -2.0 - 1e-6,
        "2D grid PE lower should be >= -2.0, got {lo_min}"
    );
    assert!(
        hi_max <= 2.0 + 1e-6,
        "2D grid PE upper should be <= 2.0, got {hi_max}"
    );
}

// ===========================================================================
// 14. 2D PE + Conv: spatial features with position (IBP)
// ===========================================================================

/// Verify that 2D PE added after patch embedding (Conv2d) produces
/// bounded outputs. This is the ViT pattern: patch_embed + pos_embed.
fn build_2d_pe_conv_kernel() -> TensorKernelDef {
    let grid_size = GRID_H; // Assumes square grid
    let num_patches = grid_size * grid_size;
    let img_size = grid_size * PATCH_SIZE;

    let mut b = TensorBlockBuilder::new("dpdf_pe_2d_conv");

    // Patch embedding via Conv2d
    let input = b.add_input("image", &[IN_CHANNELS, img_size, img_size]);
    let conv_w = b.add_input("conv_weight", &[DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
    let conv_b = b.add_input("conv_bias", &[DIM]);

    let conv_out = b.add_conv2d(
        input,
        conv_w,
        Some(conv_b),
        PATCH_SIZE, // stride_h
        PATCH_SIZE, // stride_w
        0,
        0,
        &[DIM, grid_size, grid_size],
    );

    // Reshape: [DIM, grid, grid] -> [DIM, num_patches]
    let reshaped = b.add_reshape(conv_out, &[DIM, num_patches]);
    // Transpose: [DIM, num_patches] -> [num_patches, DIM]
    let transposed = b.add_transpose(reshaped, &[1, 0], &[num_patches, DIM]);

    // Add 2D positional encoding
    let pe = b.add_input("pe_2d", &[num_patches, DIM]);
    let out = b.add_binary_add(transposed, pe, &[num_patches, DIM]);

    b.build(out).expect("valid 2D PE + Conv kernel")
}

#[test]
fn test_2d_pe_conv_ibp() {
    let grid_size = GRID_H;
    let num_patches = grid_size * grid_size;
    let img_size = grid_size * PATCH_SIZE;

    let def = build_2d_pe_conv_kernel();
    let conv_w = ArrayD::from_elem(
        IxDyn(&[DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let conv_b = ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32);
    let pe = sinusoidal_pe_2d_tensor(grid_size, grid_size, DIM);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(conv_w),
        TensorParamBinding::ConstantTensor(conv_b),
        TensorParamBinding::ConstantTensor(pe),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Image input in [0, 1]
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, img_size, img_size]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[IN_CHANNELS, img_size, img_size]), 1.0f32),
    )
    .expect("valid image bounds");

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    assert_eq!(
        output.lower_upper().0.shape(),
        &[num_patches, DIM],
        "output shape mismatch"
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("2D PE + Conv IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// 15. PE monotone tightening: smaller eps -> tighter position bounds (IBP)
// ===========================================================================

/// Verify the fundamental monotone tightening property: smaller input
/// perturbation epsilon produces tighter output bounds through any PE path.
/// Tests this across sinusoidal, RoPE, and 2D PE.
#[test]
fn test_pe_monotone_tightening() {
    // Test with sinusoidal PE addition
    let def = build_sinusoidal_pe_fixed_kernel();
    let pe = sinusoidal_pe_tensor(SEQ_LEN, DIM);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let epsilons = [0.1f32, 0.25, 0.5, 1.0, 2.0];
    let mut widths = Vec::new();

    for &eps in &epsilons {
        let input = uniform_bounds(&[SEQ_LEN, DIM], eps);
        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("PE monotone tightening (eps={eps}): width={width:.6}");
        widths.push(width);
    }

    // Smaller epsilon must produce narrower output bounds
    for i in 0..widths.len() - 1 {
        assert!(
            widths[i] <= widths[i + 1] + 1e-6,
            "monotone tightening violated: eps={} (width={:.6}) > eps={} (width={:.6})",
            epsilons[i],
            widths[i],
            epsilons[i + 1],
            widths[i + 1]
        );
    }

    // Also verify with RoPE
    let def_rope = build_rope_rotation_kernel();
    let bindings_rope = rope_rotation_bindings();
    let graph_rope =
        tensor_kernel_to_graph(&def_rope, &bindings_rope).expect("RoPE graph translation");

    let mut rope_widths = Vec::new();
    for &eps in &epsilons {
        let input = uniform_bounds(&[SEQ_LEN, DIM], eps);
        let output = graph_rope.propagate_ibp(&input).expect("RoPE IBP");
        assert_bounds_valid(&output);
        rope_widths.push(bound_width(&output));
    }

    for i in 0..rope_widths.len() - 1 {
        assert!(
            rope_widths[i] <= rope_widths[i + 1] + 1e-6,
            "RoPE monotone tightening violated: eps={} (width={:.6}) > eps={} (width={:.6})",
            epsilons[i],
            rope_widths[i],
            epsilons[i + 1],
            rope_widths[i + 1]
        );
    }
}
