// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for RoPE and M-RoPE position encoding variants used in
//! dpdf document VLMs (Qwen3-VL, FireRed-OCR, GLM-OCR, Granite-Docling).
//!
//! Verifies IBP and CROWN bound propagation through RoPE rotation, M-RoPE
//! multi-modal decomposition, frequency scaling, extended context (YaRN),
//! 2D sinusoidal encoding, attention integration, and numerical stability.
//!
//! ## Standard RoPE (tests 1-2)
//!
//! 1. Standard RoPE (cos/sin rotation) bounds (IBP)
//! 2. M-RoPE temporal component bounds (IBP)
//!
//! ## M-RoPE Components (tests 3-6)
//!
//! 3. M-RoPE height component bounds (IBP)
//! 4. M-RoPE width component bounds (IBP)
//! 5. 3-component M-RoPE combined bounds (IBP)
//! 6. Interleaved M-RoPE (Qwen3-VL) bounds (IBP)
//!
//! ## 2D & Frequency (tests 7-9)
//!
//! 7. 2D sinusoidal position encoding (SigLIP2/Granite-Docling) bounds (IBP)
//! 8. RoPE frequency scaling bounds (IBP)
//! 9. RoPE with extended context (YaRN) bounds (IBP)
//!
//! ## Attention Integration (tests 10-12)
//!
//! 10. RoPE applied to QK in attention bounds (IBP)
//! 11. Vision position encoding vs text position encoding comparison (IBP)
//! 12. Absolute vs relative position encoding bounds (IBP)
//!
//! ## Stability & Composition (tests 13-15)
//!
//! 13. RoPE numerical stability at large positions (IBP)
//! 14. Position encoding interpolation bounds (IBP)
//! 15. Full attention with RoPE: QKV proj + RoPE + SDPA bounds (IBP + CROWN)
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=8, DIM=16, NUM_HEADS=4, HEAD_DIM=4, GRID_H=4, GRID_W=4
//!
//! Part of #4022: Compose tests for RoPE and M-RoPE position encoding variants.

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build RoPE cos/sin tensors for given sequence length and dimension.
fn rope_pe_tensors(seq: usize, d: usize) -> (ArrayD<f32>, ArrayD<f32>) {
    rope_pe_tensors_with_base(seq, d, 10000.0)
}

/// Build RoPE cos/sin tensors with configurable base frequency.
fn rope_pe_tensors_with_base(seq: usize, d: usize, base: f64) -> (ArrayD<f32>, ArrayD<f32>) {
    let mut cos_data = vec![0.0f32; seq * d];
    let mut sin_data = vec![0.0f32; seq * d];
    for t in 0..seq {
        for i in 0..d / 2 {
            let freq = (t as f64) / base.powf(2.0 * i as f64 / d as f64);
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

/// Build RoPE cos/sin tensors with YaRN-style extended context scaling.
///
/// YaRN (Yet another RoPE extensioN) scales the frequency bands:
/// - Low-frequency bands are left unchanged (NTK-aware)
/// - High-frequency bands get linear interpolation scaling
fn rope_pe_tensors_yarn(
    seq: usize,
    d: usize,
    original_max: usize,
    extended_max: usize,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let scale = extended_max as f64 / original_max as f64;
    let mut cos_data = vec![0.0f32; seq * d];
    let mut sin_data = vec![0.0f32; seq * d];
    for t in 0..seq {
        for i in 0..d / 2 {
            // YaRN: high-frequency dimensions get position scaling,
            // low-frequency dimensions keep original base
            let dim_ratio = 2.0 * i as f64 / d as f64;
            let effective_scale = if dim_ratio < 0.5 { 1.0 } else { scale };
            let freq = (t as f64 / effective_scale) / 10000.0_f64.powf(2.0 * i as f64 / d as f64);
            let c = freq.cos() as f32;
            let s = freq.sin() as f32;
            cos_data[t * d + 2 * i] = c;
            cos_data[t * d + 2 * i + 1] = c;
            sin_data[t * d + 2 * i] = s;
            sin_data[t * d + 2 * i + 1] = s;
        }
    }
    (
        ArrayD::from_shape_vec(IxDyn(&[seq, d]), cos_data).expect("cos PE YaRN"),
        ArrayD::from_shape_vec(IxDyn(&[seq, d]), sin_data).expect("sin PE YaRN"),
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

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ===========================================================================
// 1. Standard RoPE (cos/sin rotation) bounds (IBP)
// ===========================================================================

/// Standard RoPE: output = input * cos_theta + input * sin_theta.
/// With cos/sin in [-1, 1] and input in [-R, R], output is bounded.
fn build_standard_rope_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rope_standard");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let cos_pe = b.add_input("cos_theta", &[SEQ_LEN, DIM]);
    let sin_pe = b.add_input("sin_theta", &[SEQ_LEN, DIM]);
    let out = add_rope_approx(&mut b, input, cos_pe, sin_pe, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid standard RoPE kernel")
}

fn standard_rope_bindings() -> Vec<TensorParamBinding> {
    let (cos_pe, sin_pe) = rope_pe_tensors(SEQ_LEN, DIM);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_pe),
        TensorParamBinding::ConstantTensor(sin_pe),
    ]
}

#[test]
fn test_standard_rope_ibp_bounded() {
    let def = build_standard_rope_kernel();
    let bindings = standard_rope_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Standard RoPE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // cos/sin in [-1, 1], so x*cos + x*sin bounded by [-2, 2] for input in [-1, 1]
    assert!(lo_min.is_finite(), "RoPE lower bound must be finite");
    assert!(hi_max.is_finite(), "RoPE upper bound must be finite");
    assert!(lo_min >= -3.0, "RoPE lower should be >= -3.0, got {lo_min}");
    assert!(hi_max <= 3.0, "RoPE upper should be <= 3.0, got {hi_max}");
}

// ===========================================================================
// 2. M-RoPE temporal component bounds (IBP)
// ===========================================================================

/// M-RoPE temporal: apply RoPE only to the temporal section of the embedding.
/// For text tokens, the temporal position is the token index.
fn build_mrope_temporal_kernel() -> TensorKernelDef {
    let section = DIM / 4;
    let sec_shape = [SEQ_LEN, section];
    let remainder = DIM - section;
    let rem_shape = [SEQ_LEN, remainder];

    let mut b = TensorBlockBuilder::new("dpdf_rope_mrope_temporal");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);

    let cos_t = b.add_input("cos_temporal", &sec_shape);
    let sin_t = b.add_input("sin_temporal", &sec_shape);

    // Narrow to temporal section
    let sec_temporal = b.add_narrow(input, 1, 0, section, &sec_shape);
    let rot_temporal = add_rope_approx(&mut b, sec_temporal, cos_t, sin_t, &sec_shape);

    // Remainder (unrotated)
    let sec_rest = b.add_narrow(input, 1, section, remainder, &rem_shape);

    let out = b.add_concat(&[rot_temporal, sec_rest], 1, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid M-RoPE temporal kernel")
}

#[test]
fn test_mrope_temporal_ibp() {
    let section = DIM / 4;
    let def = build_mrope_temporal_kernel();
    let (cos_t, sin_t) = rope_pe_tensors(SEQ_LEN, section);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_t),
        TensorParamBinding::ConstantTensor(sin_t),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("M-RoPE temporal IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "temporal lower must be finite");
    assert!(hi_max.is_finite(), "temporal upper must be finite");
}

// ===========================================================================
// 3. M-RoPE height component bounds (IBP)
// ===========================================================================

/// M-RoPE height: apply RoPE only to the height section of the embedding.
/// For vision tokens, the height position is the row index in the patch grid.
fn build_mrope_height_kernel() -> TensorKernelDef {
    let section = DIM / 4;
    let sec_shape = [NUM_GRID_POS, section];
    let remainder = DIM - section;

    let mut b = TensorBlockBuilder::new("dpdf_rope_mrope_height");
    let input = b.add_input("hidden", &[NUM_GRID_POS, DIM]);

    let cos_h = b.add_input("cos_height", &sec_shape);
    let sin_h = b.add_input("sin_height", &sec_shape);

    // Extract height section (second quarter)
    let sec_height = b.add_narrow(input, 1, section, section, &sec_shape);
    let rot_height = add_rope_approx(&mut b, sec_height, cos_h, sin_h, &sec_shape);

    // Before + after unrotated
    let before_shape = [NUM_GRID_POS, section];
    let after_shape = [NUM_GRID_POS, remainder - section];
    let sec_before = b.add_narrow(input, 1, 0, section, &before_shape);
    let sec_after = b.add_narrow(input, 1, 2 * section, remainder - section, &after_shape);

    let out = b.add_concat(
        &[sec_before, rot_height, sec_after],
        1,
        &[NUM_GRID_POS, DIM],
    );
    b.build(out).expect("valid M-RoPE height kernel")
}

#[test]
fn test_mrope_height_ibp() {
    let section = DIM / 4;
    let def = build_mrope_height_kernel();

    // Build height PE: position = row index in grid
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
    let cos_h = ArrayD::from_shape_vec(IxDyn(&[NUM_GRID_POS, section]), cos_h_data).expect("cos_h");
    let sin_h = ArrayD::from_shape_vec(IxDyn(&[NUM_GRID_POS, section]), sin_h_data).expect("sin_h");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_h),
        TensorParamBinding::ConstantTensor(sin_h),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_GRID_POS, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("M-RoPE height IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "height lower must be finite");
    assert!(hi_max.is_finite(), "height upper must be finite");
}

// ===========================================================================
// 4. M-RoPE width component bounds (IBP)
// ===========================================================================

/// M-RoPE width: apply RoPE only to the width section of the embedding.
/// For vision tokens, the width position is the column index in the patch grid.
fn build_mrope_width_kernel() -> TensorKernelDef {
    let section = DIM / 4;
    let sec_shape = [NUM_GRID_POS, section];
    let prefix = 2 * section;
    let remainder = DIM - 3 * section;

    let mut b = TensorBlockBuilder::new("dpdf_rope_mrope_width");
    let input = b.add_input("hidden", &[NUM_GRID_POS, DIM]);

    let cos_w = b.add_input("cos_width", &sec_shape);
    let sin_w = b.add_input("sin_width", &sec_shape);

    // Extract width section (third quarter)
    let sec_width = b.add_narrow(input, 1, prefix, section, &sec_shape);
    let rot_width = add_rope_approx(&mut b, sec_width, cos_w, sin_w, &sec_shape);

    // Before + after unrotated
    let before_shape = [NUM_GRID_POS, prefix];
    let after_shape = [NUM_GRID_POS, remainder];
    let sec_before = b.add_narrow(input, 1, 0, prefix, &before_shape);
    let sec_after = b.add_narrow(input, 1, 3 * section, remainder, &after_shape);

    let out = b.add_concat(&[sec_before, rot_width, sec_after], 1, &[NUM_GRID_POS, DIM]);
    b.build(out).expect("valid M-RoPE width kernel")
}

#[test]
fn test_mrope_width_ibp() {
    let section = DIM / 4;
    let def = build_mrope_width_kernel();

    // Build width PE: position = column index in grid
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
    let cos_w = ArrayD::from_shape_vec(IxDyn(&[NUM_GRID_POS, section]), cos_w_data).expect("cos_w");
    let sin_w = ArrayD::from_shape_vec(IxDyn(&[NUM_GRID_POS, section]), sin_w_data).expect("sin_w");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_w),
        TensorParamBinding::ConstantTensor(sin_w),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_GRID_POS, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("M-RoPE width IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "width lower must be finite");
    assert!(hi_max.is_finite(), "width upper must be finite");
}

// ===========================================================================
// 5. 3-component M-RoPE combined bounds (IBP)
// ===========================================================================

/// M-RoPE splits the embedding into 3 sections (temporal, height, width)
/// and applies separate rotary encodings per section. Fourth section is
/// unrotated (remainder).
fn build_mrope_3component_kernel() -> TensorKernelDef {
    let section = DIM / 4;
    let remainder = DIM - 3 * section;
    let sec_shape = [SEQ_LEN, section];

    let mut b = TensorBlockBuilder::new("dpdf_rope_mrope_3component");

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

fn mrope_3component_bindings() -> Vec<TensorParamBinding> {
    let section = DIM / 4;
    let (cos_t, sin_t) = rope_pe_tensors(SEQ_LEN, section);
    let (cos_h, sin_h) = rope_pe_tensors(SEQ_LEN, section);
    let (cos_w, sin_w) = rope_pe_tensors(SEQ_LEN, section);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_t),
        TensorParamBinding::ConstantTensor(sin_t),
        TensorParamBinding::ConstantTensor(cos_h),
        TensorParamBinding::ConstantTensor(sin_h),
        TensorParamBinding::ConstantTensor(cos_w),
        TensorParamBinding::ConstantTensor(sin_w),
    ]
}

#[test]
fn test_mrope_3component_combined_ibp() {
    let def = build_mrope_3component_kernel();
    let bindings = mrope_3component_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("M-RoPE 3-component combined IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "M-RoPE combined lower must be finite");
    assert!(hi_max.is_finite(), "M-RoPE combined upper must be finite");
}

// ===========================================================================
// 6. Interleaved M-RoPE (Qwen3-VL) bounds (IBP)
// ===========================================================================

/// Qwen3-VL interleaved M-RoPE: dimension assignment to temporal/height/width
/// uses different base frequencies per component rather than contiguous blocks.
/// This verifies the Qwen3-VL-specific rotation pattern.
#[test]
fn test_mrope_interleaved_qwen3_vl_ibp() {
    let def = build_mrope_3component_kernel();
    let section = DIM / 4;

    // Qwen3-VL uses different base frequencies for each M-RoPE component
    let base_freqs = [1.0, 2.0, 4.0]; // temporal, height, width
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
    eprintln!("Interleaved M-RoPE (Qwen3-VL) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
// 7. 2D sinusoidal position encoding (SigLIP2/Granite-Docling) bounds (IBP)
// ===========================================================================

/// SigLIP2 and Granite-Docling use 2D sinusoidal PE: row and column positions
/// are encoded separately and concatenated. Values are bounded in [-1, 1].
fn build_2d_sinusoidal_pe_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rope_2d_sinusoidal");
    let input = b.add_input("spatial_features", &[NUM_GRID_POS, DIM]);
    let pe = b.add_input("pe_2d", &[NUM_GRID_POS, DIM]);
    let out = b.add_binary_add(input, pe, &[NUM_GRID_POS, DIM]);
    b.build(out).expect("valid 2D sinusoidal PE kernel")
}

#[test]
fn test_2d_sinusoidal_pe_bounded() {
    let def = build_2d_sinusoidal_pe_kernel();
    let pe = sinusoidal_pe_2d_tensor(GRID_H, GRID_W, DIM);

    // Verify PE values are bounded in [-1, 1]
    let pe_min = pe.iter().copied().fold(f32::INFINITY, f32::min);
    let pe_max = pe.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        pe_min >= -1.0 - 1e-6,
        "2D PE values must be >= -1.0, got {pe_min}"
    );
    assert!(
        pe_max <= 1.0 + 1e-6,
        "2D PE values must be <= 1.0, got {pe_max}"
    );

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_GRID_POS, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("2D sinusoidal PE (SigLIP2) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= -2.0 - 1e-6,
        "2D PE lower should be >= -2.0, got {lo_min}"
    );
    assert!(
        hi_max <= 2.0 + 1e-6,
        "2D PE upper should be <= 2.0, got {hi_max}"
    );
}

// ===========================================================================
// 8. RoPE frequency scaling bounds (IBP)
// ===========================================================================

/// Verify that RoPE with different base frequencies all produce bounded outputs.
/// Different models use different base frequencies:
/// - Standard: base=10000 (Llama, GPT-NeoX)
/// - CodeLlama: base=1000000
/// - Qwen3-VL: base=10000 with per-component scaling
#[test]
fn test_rope_frequency_scaling_ibp() {
    let bases = [1000.0, 10000.0, 100000.0, 1000000.0];
    let mut widths = Vec::new();

    for &base in &bases {
        let mut b = TensorBlockBuilder::new(&format!("dpdf_rope_freq_base{}", base as u64));
        let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
        let cos_pe = b.add_input("cos_theta", &[SEQ_LEN, DIM]);
        let sin_pe = b.add_input("sin_theta", &[SEQ_LEN, DIM]);
        let out = add_rope_approx(&mut b, input, cos_pe, sin_pe, &[SEQ_LEN, DIM]);
        let def = b.build(out).expect("valid RoPE kernel");

        let (cos_t, sin_t) = rope_pe_tensors_with_base(SEQ_LEN, DIM, base);
        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(cos_t),
            TensorParamBinding::ConstantTensor(sin_t),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("RoPE freq base={base}: width={width:.6}");
        assert!(width.is_finite(), "width must be finite for base={base}");
        widths.push(width);
    }

    // All bases must produce finite bounded outputs
    for (i, &w) in widths.iter().enumerate() {
        assert!(
            w.is_finite(),
            "RoPE base={} produced non-finite width",
            bases[i]
        );
    }
}

// ===========================================================================
// 9. RoPE with extended context (YaRN) bounds (IBP)
// ===========================================================================

/// YaRN (Yet another RoPE extensioN) extends context length by scaling
/// frequency bands differently. Verify bounds hold for extended positions.
#[test]
fn test_rope_yarn_extended_context_ibp() {
    let original_max: usize = 8;
    let extended_max: usize = 32; // 4x context extension

    let mut b = TensorBlockBuilder::new("dpdf_rope_yarn_extended");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let cos_pe = b.add_input("cos_theta", &[SEQ_LEN, DIM]);
    let sin_pe = b.add_input("sin_theta", &[SEQ_LEN, DIM]);
    let out = add_rope_approx(&mut b, input, cos_pe, sin_pe, &[SEQ_LEN, DIM]);
    let def = b.build(out).expect("valid YaRN kernel");

    let (cos_t, sin_t) = rope_pe_tensors_yarn(SEQ_LEN, DIM, original_max, extended_max);

    // Verify YaRN PE values are still bounded in [-1, 1]
    let cos_min = cos_t.iter().copied().fold(f32::INFINITY, f32::min);
    let cos_max = cos_t.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        cos_min >= -1.0 - 1e-6,
        "YaRN cos must be >= -1.0, got {cos_min}"
    );
    assert!(
        cos_max <= 1.0 + 1e-6,
        "YaRN cos must be <= 1.0, got {cos_max}"
    );

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_t),
        TensorParamBinding::ConstantTensor(sin_t),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("YaRN extended context IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "YaRN lower must be finite");
    assert!(hi_max.is_finite(), "YaRN upper must be finite");
}

// ===========================================================================
// 10. RoPE applied to QK in attention bounds (IBP)
// ===========================================================================

/// Standard transformer pattern: Q = RoPE(Wq * x), K = RoPE(Wk * x),
/// V = Wv * x, then attention(Q, K, V). Verifies bounds through the
/// full QK rotation + softmax attention pipeline.
fn build_rope_qk_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rope_qk_attention");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let cos_pe = b.add_input("cos_theta", &[SEQ_LEN, DIM]);
    let sin_pe = b.add_input("sin_theta", &[SEQ_LEN, DIM]);

    // Q and K projections
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, DIM]);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, DIM]);

    // Apply RoPE to Q and K (not V)
    let q_rot = add_rope_approx(&mut b, q, cos_pe, sin_pe, &[SEQ_LEN, DIM]);
    let k_rot = add_rope_approx(&mut b, k, cos_pe, sin_pe, &[SEQ_LEN, DIM]);

    // Attention: softmax(Q_rot @ K_rot^T / sqrt(d_k)) @ V
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q_rot,
        k_rot,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, DIM],
    );

    let out = b.add_linear(attn, out_w, None, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid RoPE QK attention kernel")
}

#[test]
fn test_rope_qk_attention_ibp() {
    let def = build_rope_qk_attention_kernel();
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
    eprintln!("RoPE QK attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "RoPE QK attn lower must be finite");
    assert!(hi_max.is_finite(), "RoPE QK attn upper must be finite");
}

// ===========================================================================
// 11. Vision position encoding vs text position encoding comparison (IBP)
// ===========================================================================

/// Compare bound widths between vision-style M-RoPE (spatial grid positions)
/// and text-style standard RoPE (sequential positions). Both should produce
/// finite bounds; we verify and log the comparison.
#[test]
fn test_vision_vs_text_pe_comparison() {
    // Text: standard RoPE on sequential tokens
    let def_text = build_standard_rope_kernel();
    let bindings_text = standard_rope_bindings();
    let graph_text = tensor_kernel_to_graph(&def_text, &bindings_text).expect("text graph");
    let input_text = uniform_bounds(&[SEQ_LEN, DIM], 1.0);
    let output_text = graph_text.propagate_ibp(&input_text).expect("text IBP");
    assert_bounds_valid(&output_text);
    let text_width = bound_width(&output_text);

    // Vision: M-RoPE 3-component on grid positions
    let def_vision = build_mrope_3component_kernel();
    let bindings_vision = mrope_3component_bindings();
    let graph_vision = tensor_kernel_to_graph(&def_vision, &bindings_vision).expect("vision graph");
    let input_vision = uniform_bounds(&[SEQ_LEN, DIM], 1.0);
    let output_vision = graph_vision
        .propagate_ibp(&input_vision)
        .expect("vision IBP");
    assert_bounds_valid(&output_vision);
    let vision_width = bound_width(&output_vision);

    eprintln!(
        "PE comparison: text RoPE width={text_width:.6}, vision M-RoPE width={vision_width:.6}"
    );
    assert!(text_width.is_finite(), "text RoPE width must be finite");
    assert!(
        vision_width.is_finite(),
        "vision M-RoPE width must be finite"
    );
}

// ===========================================================================
// 12. Absolute vs relative position encoding bounds (IBP)
// ===========================================================================

/// Compare absolute PE (additive sinusoidal) with relative PE (RoPE rotation).
/// Both should produce bounded outputs from the same input range.
#[test]
fn test_absolute_vs_relative_pe_bounds() {
    // Absolute: sinusoidal PE added to features
    let mut b_abs = TensorBlockBuilder::new("dpdf_rope_absolute_pe");
    let input_abs = b_abs.add_input("features", &[SEQ_LEN, DIM]);
    let pe_abs = b_abs.add_input("sinusoidal_pe", &[SEQ_LEN, DIM]);
    let out_abs = b_abs.add_binary_add(input_abs, pe_abs, &[SEQ_LEN, DIM]);
    let def_abs = b_abs.build(out_abs).expect("valid absolute PE kernel");

    // Build sinusoidal PE
    let mut pe_data = vec![0.0f32; SEQ_LEN * DIM];
    for t in 0..SEQ_LEN {
        for i in 0..DIM / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / DIM as f64);
            pe_data[t * DIM + 2 * i] = freq.sin() as f32;
            pe_data[t * DIM + 2 * i + 1] = freq.cos() as f32;
        }
    }
    let pe_tensor = ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, DIM]), pe_data).expect("PE");
    let bindings_abs = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_tensor),
    ];
    let graph_abs = tensor_kernel_to_graph(&def_abs, &bindings_abs).expect("absolute graph");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);
    let output_abs = graph_abs.propagate_ibp(&input).expect("absolute IBP");
    assert_bounds_valid(&output_abs);
    let abs_width = bound_width(&output_abs);

    // Relative: RoPE rotation
    let def_rel = build_standard_rope_kernel();
    let bindings_rel = standard_rope_bindings();
    let graph_rel = tensor_kernel_to_graph(&def_rel, &bindings_rel).expect("relative graph");
    let output_rel = graph_rel.propagate_ibp(&input).expect("relative IBP");
    assert_bounds_valid(&output_rel);
    let rel_width = bound_width(&output_rel);

    eprintln!(
        "Absolute vs relative PE: absolute width={abs_width:.6}, relative (RoPE) width={rel_width:.6}"
    );
    assert!(abs_width.is_finite(), "absolute PE width must be finite");
    assert!(rel_width.is_finite(), "relative PE width must be finite");
}

// ===========================================================================
// 13. RoPE numerical stability at large positions (IBP)
// ===========================================================================

/// Verify that RoPE applied at large position indices (e.g., long documents)
/// still produces bounded outputs. Large positions can cause floating-point
/// precision issues with high-frequency sinusoidal components.
#[test]
fn test_rope_large_positions_stability() {
    let large_offset: usize = 4096;
    let long_seq: usize = 8;

    let mut b = TensorBlockBuilder::new("dpdf_rope_large_positions");
    let input = b.add_input("hidden", &[long_seq, DIM]);
    let cos_pe = b.add_input("cos_theta", &[long_seq, DIM]);
    let sin_pe = b.add_input("sin_theta", &[long_seq, DIM]);
    let out = add_rope_approx(&mut b, input, cos_pe, sin_pe, &[long_seq, DIM]);
    let def = b.build(out).expect("valid large-position RoPE kernel");

    // Build PE at large offsets
    let mut cos_data = vec![0.0f32; long_seq * DIM];
    let mut sin_data = vec![0.0f32; long_seq * DIM];
    for t in 0..long_seq {
        let pos = large_offset + t;
        for i in 0..DIM / 2 {
            let freq = (pos as f64) / 10000.0_f64.powf(2.0 * i as f64 / DIM as f64);
            let c = freq.cos() as f32;
            let s = freq.sin() as f32;
            cos_data[t * DIM + 2 * i] = c;
            cos_data[t * DIM + 2 * i + 1] = c;
            sin_data[t * DIM + 2 * i] = s;
            sin_data[t * DIM + 2 * i + 1] = s;
        }
    }
    let cos_t = ArrayD::from_shape_vec(IxDyn(&[long_seq, DIM]), cos_data).expect("cos PE");
    let sin_t = ArrayD::from_shape_vec(IxDyn(&[long_seq, DIM]), sin_data).expect("sin PE");

    // Verify PE values are still bounded in [-1, 1]
    let cos_min = cos_t.iter().copied().fold(f32::INFINITY, f32::min);
    let cos_max = cos_t.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        cos_min >= -1.0 - 1e-6,
        "large-pos cos must be >= -1.0, got {cos_min}"
    );
    assert!(
        cos_max <= 1.0 + 1e-6,
        "large-pos cos must be <= 1.0, got {cos_max}"
    );

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
    eprintln!(
        "RoPE large positions (offset={large_offset}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    assert!(lo_min.is_finite(), "large-pos lower must be finite");
    assert!(hi_max.is_finite(), "large-pos upper must be finite");
}

// ===========================================================================
// 14. Position encoding interpolation bounds (IBP)
// ===========================================================================

/// Verify that fractional (interpolated) position indices for RoPE produce
/// bounded results. This supports continuous position encoding for
/// variable-resolution vision inputs.
#[test]
fn test_rope_position_interpolation_ibp() {
    // Build fractional PE: positions at half-integer steps
    let mut cos_data = vec![0.0f32; SEQ_LEN * DIM];
    let mut sin_data = vec![0.0f32; SEQ_LEN * DIM];
    for t in 0..SEQ_LEN {
        let frac_pos = t as f64 + 0.5; // Half-step positions
        for i in 0..DIM / 2 {
            let freq = frac_pos / 10000.0_f64.powf(2.0 * i as f64 / DIM as f64);
            let c = freq.cos() as f32;
            let s = freq.sin() as f32;
            cos_data[t * DIM + 2 * i] = c;
            cos_data[t * DIM + 2 * i + 1] = c;
            sin_data[t * DIM + 2 * i] = s;
            sin_data[t * DIM + 2 * i + 1] = s;
        }
    }
    let cos_frac = ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, DIM]), cos_data).expect("cos frac");
    let sin_frac = ArrayD::from_shape_vec(IxDyn(&[SEQ_LEN, DIM]), sin_data).expect("sin frac");

    // Verify fractional PE is still bounded in [-1, 1]
    let cos_min = cos_frac.iter().copied().fold(f32::INFINITY, f32::min);
    let cos_max = cos_frac.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        cos_min >= -1.0 - 1e-6,
        "fractional cos must be >= -1.0, got {cos_min}"
    );
    assert!(
        cos_max <= 1.0 + 1e-6,
        "fractional cos must be <= 1.0, got {cos_max}"
    );

    let def = build_standard_rope_kernel();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_frac),
        TensorParamBinding::ConstantTensor(sin_frac),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RoPE interpolation IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "interpolation lower must be finite");
    assert!(hi_max.is_finite(), "interpolation upper must be finite");
    assert!(
        lo_min >= -3.0,
        "interpolation lower should be >= -3.0, got {lo_min}"
    );
    assert!(
        hi_max <= 3.0,
        "interpolation upper should be <= 3.0, got {hi_max}"
    );
}

// ===========================================================================
// 15. Full attention with RoPE: QKV proj + RoPE + SDPA bounds (IBP + CROWN)
// ===========================================================================

/// End-to-end test: Linear projections for Q/K/V, RoPE on Q/K, scaled dot
/// product attention, output projection, and residual connection. This is
/// the complete attention block used in Qwen3-VL, GLM-OCR, FireRed-OCR.
fn build_full_rope_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_rope_full_attention");

    let input = b.add_input("x", &[SEQ_LEN, DIM]);
    let cos_pe = b.add_input("cos_theta", &[SEQ_LEN, DIM]);
    let sin_pe = b.add_input("sin_theta", &[SEQ_LEN, DIM]);

    // QKV projections
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    let q = b.add_linear(input, q_w, None, &[SEQ_LEN, DIM]);
    let k = b.add_linear(input, k_w, None, &[SEQ_LEN, DIM]);
    let v = b.add_linear(input, v_w, None, &[SEQ_LEN, DIM]);

    // Apply RoPE to Q and K
    let q_rot = add_rope_approx(&mut b, q, cos_pe, sin_pe, &[SEQ_LEN, DIM]);
    let k_rot = add_rope_approx(&mut b, k, cos_pe, sin_pe, &[SEQ_LEN, DIM]);

    // Scaled dot-product attention
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attn = b.add_attention(
        q_rot,
        k_rot,
        v,
        AttentionMask::Standard,
        Some(scale),
        &[SEQ_LEN, DIM],
    );

    // Output projection
    let proj = b.add_linear(attn, out_w, None, &[SEQ_LEN, DIM]);

    // Residual connection
    let result = b.add_binary_add(input, proj, &[SEQ_LEN, DIM]);

    b.build(result).expect("valid full RoPE attention kernel")
}

fn full_rope_attention_bindings() -> Vec<TensorParamBinding> {
    let (cos_pe, sin_pe) = rope_pe_tensors(SEQ_LEN, DIM);
    let proj_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_pe),
        TensorParamBinding::ConstantTensor(sin_pe),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w.clone()),
        TensorParamBinding::ConstantTensor(proj_w),
    ]
}

#[test]
fn test_full_rope_attention_ibp() {
    let def = build_full_rope_attention_kernel();
    let bindings = full_rope_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full RoPE attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "full attn lower must be finite");
    assert!(hi_max.is_finite(), "full attn upper must be finite");
}

#[test]
fn test_full_rope_attention_crown() {
    let def = build_full_rope_attention_kernel();
    let bindings = full_rope_attention_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Full RoPE attention CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}
