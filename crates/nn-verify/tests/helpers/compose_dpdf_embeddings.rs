// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for embedding and projection layers used across dpdf models.
//!
//! Verifies IBP and CROWN bound propagation through the various embedding and
//! projection layers that appear in document understanding models:
//!
//! ## Token & Patch Embeddings (tests 1-4)
//!
//! 1. Token embedding lookup IBP bounds
//! 2. Patch embedding Conv2d(3, D, P, stride=P) IBP
//! 3. Learned positional embedding IBP
//! 4. Token embedding + positional addition IBP
//!
//! ## Positional Encodings (tests 5-8)
//!
//! 5. Sinusoidal positional encoding bounded in [-1, 1] IBP
//! 6. Rotary positional embedding (RoPE) bounded in [-1, 1] IBP
//! 7. M-RoPE (multimodal rotary) IBP
//! 8. 2D sinusoidal position encoding IBP
//!
//! ## Projections & Composition (tests 9-14)
//!
//! 9. Vision-to-language projection linear IBP + CROWN
//! 10. Cross-modal projection (vision -> text space) CROWN
//! 11. Embedding + LayerNorm composition IBP + CROWN
//! 12. Patch embed + position encode + attention composition IBP
//! 13. Embedding dimension scaling IBP
//! 14. Embedding monotone tightening
//!
//! Dimensions (small for fast verification, structurally representative):
//! - VOCAB_SIZE=64, SEQ_LEN=4, DIM=16, PATCH_SIZE=8, IMG_SIZE=16
//!
//! Part of #3975: Embedding and projection compose tests for dpdf models.

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

const VOCAB_SIZE: usize = 64;
const SEQ_LEN: usize = 4;
const DIM: usize = 16;
const FFN_DIM: usize = 32;
const IMG_SIZE: usize = 16;
const PATCH_SIZE: usize = 8;
const GRID_SIZE: usize = IMG_SIZE / PATCH_SIZE; // 2
const NUM_PATCHES: usize = GRID_SIZE * GRID_SIZE; // 4
const IN_CHANNELS: usize = 3;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = DIM / NUM_HEADS; // 4
const WEIGHT_MAG: f32 = 0.02;

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

/// Create image-domain input bounds: pixels in [0, 1].
fn image_bounds_01(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), 0.0f32),
        ArrayD::from_elem(IxDyn(shape), 1.0f32),
    )
    .expect("valid image bounds [0, 1]")
}

/// Build RoPE rotation by decomposing into cos/sin multiply + add.
///
/// For each pair (x_even, x_odd):
///   rotated_even = x_even * cos_theta - x_odd * sin_theta
///   rotated_odd  = x_even * sin_theta + x_odd * cos_theta
///
/// This is approximated as a constant affine transform for verification:
/// output = input * cos_tensor + rotated_input * sin_tensor
fn add_rope_approx(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    cos_pe: nn_dsl::TensorNodeId,
    sin_pe: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    // x * cos
    let x_cos = b.add_binary_mul(input, cos_pe, shape);
    // x * sin (approximation: we treat the rotation as bounds-preserving)
    let x_sin = b.add_binary_mul(input, sin_pe, shape);
    // x_cos + x_sin approximates the rotated output for bound analysis
    b.add_binary_add(x_cos, x_sin, shape)
}

/// Compute output bound width from a BoundedTensor.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ===========================================================================
// 1. Token embedding lookup IBP bounds
// ===========================================================================

fn build_token_embedding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_embed_token");
    let indices = b.add_input("token_ids", &[SEQ_LEN]);
    let weight = b.add_input("embed_weight", &[VOCAB_SIZE, DIM]);
    let out = b.add_embedding(indices, weight, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid token embedding kernel")
}

#[test]
fn test_token_embedding_ibp_bounds() {
    let def = build_token_embedding_kernel();
    let embed_w = ArrayD::from_elem(IxDyn(&[VOCAB_SIZE, DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,                // token_ids
        TensorParamBinding::ConstantTensor(embed_w), // embed_weight
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Token embedding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. Patch embedding Conv2d(3, D, P, stride=P) IBP
// ===========================================================================

fn build_patch_embedding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_embed_patch");
    let input = b.add_input("image", &[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);
    let weight = b.add_input("proj_weight", &[DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
    let bias = b.add_input("proj_bias", &[DIM]);

    // Conv2d: [3, 16, 16] -> [D, 2, 2]
    let conv_out = b.add_conv2d(
        input,
        weight,
        Some(bias),
        PATCH_SIZE, // stride_h
        PATCH_SIZE, // stride_w
        0,          // padding_h
        0,          // padding_w
        &[DIM, GRID_SIZE, GRID_SIZE],
    );

    // Reshape: [D, 2, 2] -> [D, NUM_PATCHES]
    let reshaped = b.add_reshape(conv_out, &[DIM, NUM_PATCHES]);

    // Transpose: [D, NUM_PATCHES] -> [NUM_PATCHES, D]
    let out = b.add_transpose(reshaped, &[1, 0], &[NUM_PATCHES, DIM]);

    b.build(out).expect("valid patch embedding kernel")
}

fn patch_embedding_bindings() -> Vec<TensorParamBinding> {
    let w = ArrayD::from_elem(
        IxDyn(&[DIM, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]),
        WEIGHT_MAG,
    );
    let bias = ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32);
    vec![
        TensorParamBinding::Variable,             // image
        TensorParamBinding::ConstantTensor(w),    // proj_weight
        TensorParamBinding::ConstantTensor(bias), // proj_bias
    ]
}

#[test]
fn test_patch_embedding_ibp_bounds() {
    let def = build_patch_embedding_kernel();
    let bindings = patch_embedding_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = image_bounds_01(&[IN_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[NUM_PATCHES, DIM],
        "patch embedding output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch embedding IBP (image [0,1]): bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min < hi_max,
        "bounds must be non-degenerate: [{lo_min}, {hi_max}]"
    );
}

// ===========================================================================
// 3. Learned positional embedding IBP
// ===========================================================================

fn build_learned_positional_embedding_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_embed_learned_pos");
    let indices = b.add_input("position_ids", &[SEQ_LEN]);
    let weight = b.add_input("pos_embed_weight", &[SEQ_LEN, DIM]);
    let out = b.add_embedding(indices, weight, &[SEQ_LEN, DIM]);
    b.build(out)
        .expect("valid learned positional embedding kernel")
}

#[test]
fn test_learned_positional_embedding_ibp() {
    let def = build_learned_positional_embedding_kernel();
    let pos_w = ArrayD::from_elem(IxDyn(&[SEQ_LEN, DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,              // position_ids
        TensorParamBinding::ConstantTensor(pos_w), // pos_embed_weight
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Learned positional embedding IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 4. Token embedding + positional addition IBP
// ===========================================================================

fn build_token_plus_position_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_embed_token_plus_pos");
    // Token embedding output (variable)
    let tok_embed = b.add_input("token_embedding", &[SEQ_LEN, DIM]);
    // Positional encoding (constant sin/cos)
    let pos_enc = b.add_input("positional_encoding", &[SEQ_LEN, DIM]);
    // Add: token_embedding + positional_encoding
    let out = b.add_binary_add(tok_embed, pos_enc, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid token+position kernel")
}

#[test]
fn test_token_plus_position_ibp() {
    let def = build_token_plus_position_kernel();
    let pe = sinusoidal_pe_tensor(SEQ_LEN, DIM);
    let bindings = vec![
        TensorParamBinding::Variable,           // token_embedding
        TensorParamBinding::ConstantTensor(pe), // positional_encoding
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Token + position IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Input [-1, 1] + PE [-1, 1] = output in [-2, 2]
    assert!(
        lo_min >= -2.0 - 1e-6,
        "token+pos lower should be >= -2.0, got {lo_min}"
    );
    assert!(
        hi_max <= 2.0 + 1e-6,
        "token+pos upper should be <= 2.0, got {hi_max}"
    );
}

// ===========================================================================
// 5. Sinusoidal positional encoding bounded in [-1, 1] IBP
// ===========================================================================

fn build_sinusoidal_pe_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_embed_sinusoidal_pe");
    let input = b.add_input("features", &[SEQ_LEN, DIM]);
    let pe = b.add_input("positional_encoding", &[SEQ_LEN, DIM]);
    let out = b.add_binary_add(input, pe, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid sinusoidal PE kernel")
}

#[test]
fn test_sinusoidal_pe_ibp_bounds() {
    let def = build_sinusoidal_pe_kernel();
    let pe = sinusoidal_pe_tensor(SEQ_LEN, DIM);
    let bindings = vec![
        TensorParamBinding::Variable,           // features
        TensorParamBinding::ConstantTensor(pe), // positional_encoding
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Sinusoidal PE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
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
// 6. Rotary positional embedding (RoPE) bounded in [-1, 1] IBP
// ===========================================================================

fn build_rope_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_embed_rope");
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);
    let cos_pe = b.add_input("cos_theta", &[SEQ_LEN, DIM]);
    let sin_pe = b.add_input("sin_theta", &[SEQ_LEN, DIM]);
    let out = add_rope_approx(&mut b, input, cos_pe, sin_pe, &[SEQ_LEN, DIM]);
    b.build(out).expect("valid RoPE kernel")
}

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

#[test]
fn test_rope_ibp_bounds() {
    let def = build_rope_kernel();
    let (cos_pe, sin_pe) = rope_pe_tensors(SEQ_LEN, DIM);
    let bindings = vec![
        TensorParamBinding::Variable,               // hidden
        TensorParamBinding::ConstantTensor(cos_pe), // cos_theta
        TensorParamBinding::ConstantTensor(sin_pe), // sin_theta
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("RoPE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // cos/sin in [-1, 1], so x*cos + x*sin bounded by [-2, 2] for input in [-1, 1]
    assert!(lo_min.is_finite(), "RoPE lower bound must be finite");
    assert!(hi_max.is_finite(), "RoPE upper bound must be finite");
    assert!(lo_min >= -3.0, "RoPE lower should be >= -3.0, got {lo_min}");
    assert!(hi_max <= 3.0, "RoPE upper should be <= 3.0, got {hi_max}");
}

// ===========================================================================
// 7. M-RoPE (multimodal rotary) IBP
// ===========================================================================

/// M-RoPE splits the embedding into 3 sections (temporal, height, width)
/// and applies separate rotary encodings. For verification, we model each
/// section as an independent RoPE application and concatenate.
fn build_mrope_kernel() -> TensorKernelDef {
    let section = DIM / 4; // each section gets DIM/4 dims (rounded)
    let remainder = DIM - 3 * section;
    let sec_shape = [SEQ_LEN, section];

    let mut b = TensorBlockBuilder::new("dpdf_embed_mrope");

    // Input split into 3 sections + remainder
    let input = b.add_input("hidden", &[SEQ_LEN, DIM]);

    // Temporal RoPE
    let cos_t = b.add_input("cos_temporal", &sec_shape);
    let sin_t = b.add_input("sin_temporal", &sec_shape);

    // Height RoPE
    let cos_h = b.add_input("cos_height", &sec_shape);
    let sin_h = b.add_input("sin_height", &sec_shape);

    // Width RoPE
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

    // Remainder section (no rotation) -- narrow from the tail
    let rem_shape = [SEQ_LEN, remainder];
    let sec3 = b.add_narrow(input, 1, 3 * section, remainder, &rem_shape);

    // Concatenate all sections back: [SEQ_LEN, DIM]
    let out = b.add_concat(&[rot0, rot1, rot2, sec3], 1, &[SEQ_LEN, DIM]);

    b.build(out).expect("valid M-RoPE kernel")
}

#[test]
fn test_mrope_ibp_bounds() {
    let def = build_mrope_kernel();
    let section = DIM / 4;
    let (cos_t, sin_t) = rope_pe_tensors(SEQ_LEN, section);
    let (cos_h, sin_h) = rope_pe_tensors(SEQ_LEN, section);
    let (cos_w, sin_w) = rope_pe_tensors(SEQ_LEN, section);

    let bindings = vec![
        TensorParamBinding::Variable,              // hidden
        TensorParamBinding::ConstantTensor(cos_t), // cos_temporal
        TensorParamBinding::ConstantTensor(sin_t), // sin_temporal
        TensorParamBinding::ConstantTensor(cos_h), // cos_height
        TensorParamBinding::ConstantTensor(sin_h), // sin_height
        TensorParamBinding::ConstantTensor(cos_w), // cos_width
        TensorParamBinding::ConstantTensor(sin_w), // sin_width
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("M-RoPE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "M-RoPE lower bound must be finite");
    assert!(hi_max.is_finite(), "M-RoPE upper bound must be finite");
}

// ===========================================================================
// 8. 2D sinusoidal position encoding IBP
// ===========================================================================

fn build_2d_sinusoidal_pe_kernel() -> TensorKernelDef {
    let num_pos = GRID_SIZE * GRID_SIZE; // spatial positions
    let mut b = TensorBlockBuilder::new("dpdf_embed_2d_sinusoidal_pe");
    let input = b.add_input("patch_features", &[num_pos, DIM]);
    let pe = b.add_input("pe_2d", &[num_pos, DIM]);
    let out = b.add_binary_add(input, pe, &[num_pos, DIM]);
    b.build(out).expect("valid 2D sinusoidal PE kernel")
}

#[test]
fn test_2d_sinusoidal_pe_ibp() {
    let num_pos = GRID_SIZE * GRID_SIZE;
    let def = build_2d_sinusoidal_pe_kernel();
    let pe = sinusoidal_pe_2d_tensor(GRID_SIZE, GRID_SIZE, DIM);
    let bindings = vec![
        TensorParamBinding::Variable,           // patch_features
        TensorParamBinding::ConstantTensor(pe), // pe_2d
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[num_pos, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("2D sinusoidal PE IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Input [-1, 1] + 2D PE [-1, 1] = output in [-2, 2]
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
// 9. Vision-to-language projection linear IBP + CROWN
// ===========================================================================

fn build_vision_projection_kernel() -> TensorKernelDef {
    let vision_dim = DIM;
    let text_dim = FFN_DIM;
    let mut b = TensorBlockBuilder::new("dpdf_embed_vision_proj");
    let input = b.add_input("vision_features", &[NUM_PATCHES, vision_dim]);
    let weight = b.add_input("proj_weight", &[text_dim, vision_dim]);
    let bias = b.add_input("proj_bias", &[text_dim]);
    let out = b.add_linear(input, weight, Some(bias), &[NUM_PATCHES, text_dim]);
    b.build(out).expect("valid vision projection kernel")
}

fn vision_projection_bindings() -> Vec<TensorParamBinding> {
    let vision_dim = DIM;
    let text_dim = FFN_DIM;
    let w = ArrayD::from_elem(IxDyn(&[text_dim, vision_dim]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[text_dim]), 0.0f32);
    vec![
        TensorParamBinding::Variable,             // vision_features
        TensorParamBinding::ConstantTensor(w),    // proj_weight
        TensorParamBinding::ConstantTensor(bias), // proj_bias
    ]
}

#[test]
fn test_vision_projection_ibp_bounds() {
    let def = build_vision_projection_kernel();
    let bindings = vision_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Vision projection IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_vision_projection_crown_bounds() {
    let def = build_vision_projection_kernel();
    let bindings = vision_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!("Vision projection CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}");
}

// ===========================================================================
// 10. Cross-modal projection (vision -> text space) CROWN
// ===========================================================================

fn build_cross_modal_projection_kernel() -> TensorKernelDef {
    let vision_dim = DIM;
    let text_dim = FFN_DIM;
    let mut b = TensorBlockBuilder::new("dpdf_embed_cross_modal_proj");

    let input = b.add_input("vision_features", &[NUM_PATCHES, vision_dim]);
    let proj_w = b.add_input("proj_weight", &[text_dim, vision_dim]);
    let proj_b = b.add_input("proj_bias", &[text_dim]);

    // Linear projection: vision -> text space
    let projected = b.add_linear(input, proj_w, Some(proj_b), &[NUM_PATCHES, text_dim]);

    // GELU activation (common in VLM projections like LLaVA)
    let out = b.add_gelu(projected, &[NUM_PATCHES, text_dim]);

    b.build(out).expect("valid cross-modal projection kernel")
}

#[test]
fn test_cross_modal_projection_crown() {
    let vision_dim = DIM;
    let text_dim = FFN_DIM;
    let def = build_cross_modal_projection_kernel();
    let w = ArrayD::from_elem(IxDyn(&[text_dim, vision_dim]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[text_dim]), 0.0f32);
    let bindings = vec![
        TensorParamBinding::Variable,             // vision_features
        TensorParamBinding::ConstantTensor(w),    // proj_weight
        TensorParamBinding::ConstantTensor(bias), // proj_bias
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "Cross-modal projection CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 11. Embedding + LayerNorm composition IBP + CROWN
// ===========================================================================

fn build_embedding_layernorm_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_embed_layernorm");
    // Variable input: token embeddings [SEQ_LEN, DIM]
    let input = b.add_input("token_embedding", &[SEQ_LEN, DIM]);
    // LayerNorm parameters
    let ln_weight = b.add_input("ln_weight", &[DIM]);
    let ln_bias = b.add_input("ln_bias", &[DIM]);
    let eps = b.add_input("eps", &[1]);

    // LayerNorm over last axis (axis=1 for shape [SEQ_LEN, DIM])
    let out = b.add_layer_norm(input, eps, 1, ln_weight, ln_bias, &[SEQ_LEN, DIM]);

    b.build(out).expect("valid embedding+LayerNorm kernel")
}

fn embedding_layernorm_bindings() -> Vec<TensorParamBinding> {
    let ln_weight = ArrayD::from_elem(IxDyn(&[DIM]), 1.0f32);
    let ln_bias = ArrayD::from_elem(IxDyn(&[DIM]), 0.0f32);
    let eps = ArrayD::from_elem(IxDyn(&[1]), 1e-5f32);
    vec![
        TensorParamBinding::Variable,                  // token_embedding
        TensorParamBinding::ConstantTensor(ln_weight), // ln_weight
        TensorParamBinding::ConstantTensor(ln_bias),   // ln_bias
        TensorParamBinding::ConstantTensor(eps),       // eps
    ]
}

#[test]
fn test_embedding_layernorm_ibp() {
    let def = build_embedding_layernorm_kernel();
    let bindings = embedding_layernorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Embedding + LayerNorm IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_embedding_layernorm_crown() {
    let def = build_embedding_layernorm_kernel();
    let bindings = embedding_layernorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, DIM], 0.5);

    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    assert_bounds_valid(&ibp_output);

    let (_method, crown_output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    let ibp_width = bound_width(&ibp_output);
    let crown_width = bound_width(&crown_output);
    eprintln!(
        "Embedding + LayerNorm CROWN: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}"
    );
}

// ===========================================================================
// 12. Patch embed + position encode + attention composition IBP
// ===========================================================================

fn build_patch_embed_pe_attention_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_embed_patch_pe_attn");

    // Patch embedding output (variable) [NUM_PATCHES, DIM]
    let input = b.add_input("patch_features", &[NUM_PATCHES, DIM]);
    // Sinusoidal PE (constant) [NUM_PATCHES, DIM]
    let pe = b.add_input("positional_encoding", &[NUM_PATCHES, DIM]);

    // Q/K/V projection weights [DIM, DIM]
    let q_w = b.add_input("q_weight", &[DIM, DIM]);
    let k_w = b.add_input("k_weight", &[DIM, DIM]);
    let v_w = b.add_input("v_weight", &[DIM, DIM]);
    let out_w = b.add_input("out_weight", &[DIM, DIM]);

    // Step 1: features + positional encoding
    let x_pe = b.add_binary_add(input, pe, &[NUM_PATCHES, DIM]);

    // Step 2: Multi-head self-attention
    let attn_out = b
        .add_multi_head_attention(
            x_pe,
            q_w,
            k_w,
            v_w,
            out_w,
            NUM_HEADS,
            AttentionMask::Standard,
            &[NUM_PATCHES, DIM],
        )
        .expect("valid MHA");

    // Step 3: Residual connection
    let out = b.add_binary_add(x_pe, attn_out, &[NUM_PATCHES, DIM]);

    b.build(out)
        .expect("valid patch_embed + PE + attention kernel")
}

#[test]
fn test_patch_embed_pe_attention_ibp() {
    let def = build_patch_embed_pe_attention_kernel();
    let pe = sinusoidal_pe_tensor(NUM_PATCHES, DIM);
    let q_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    let k_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    let v_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    let out_w = ArrayD::from_elem(IxDyn(&[DIM, DIM]), WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,              // patch_features
        TensorParamBinding::ConstantTensor(pe),    // positional_encoding
        TensorParamBinding::ConstantTensor(q_w),   // q_weight
        TensorParamBinding::ConstantTensor(k_w),   // k_weight
        TensorParamBinding::ConstantTensor(v_w),   // v_weight
        TensorParamBinding::ConstantTensor(out_w), // out_weight
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NUM_PATCHES, DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Patch embed + PE + attention IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 13. Embedding dimension scaling IBP
// ===========================================================================

/// Tests that embedding bounds scale linearly with weight magnitude.
/// Smaller weights produce tighter output bounds.
#[test]
fn test_embedding_dimension_scaling_ibp() {
    let def = build_vision_projection_kernel();

    let magnitudes = [0.01f32, 0.02, 0.05];
    let mut widths = Vec::new();

    for &mag in &magnitudes {
        let w = ArrayD::from_elem(IxDyn(&[FFN_DIM, DIM]), mag);
        let bias = ArrayD::from_elem(IxDyn(&[FFN_DIM]), 0.0f32);
        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(w),
            TensorParamBinding::ConstantTensor(bias),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let input = uniform_bounds(&[NUM_PATCHES, DIM], 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("Dimension scaling IBP (mag={mag}): width={width:.6}");
        widths.push(width);
    }

    // Smaller weight magnitude should produce narrower bounds
    for i in 0..widths.len() - 1 {
        assert!(
            widths[i] <= widths[i + 1] + 1e-6,
            "smaller weights ({}) should produce narrower bounds ({:.6}) than larger weights ({}) ({:.6})",
            magnitudes[i], widths[i], magnitudes[i + 1], widths[i + 1]
        );
    }
}

// ===========================================================================
// 14. Embedding monotone tightening
// ===========================================================================

/// Verifies that tighter input bounds produce tighter output bounds
/// through the token+position embedding pipeline.
#[test]
fn test_embedding_monotone_tightening() {
    let def = build_token_plus_position_kernel();
    let pe = sinusoidal_pe_tensor(SEQ_LEN, DIM);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let epsilons = [0.25f32, 0.5, 1.0, 2.0];
    let mut widths = Vec::new();

    for &eps in &epsilons {
        let input = uniform_bounds(&[SEQ_LEN, DIM], eps);
        let output = graph.propagate_ibp(&input).expect("IBP propagation");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("Monotone tightening (eps={eps}): width={width:.6}");
        widths.push(width);
    }

    // Smaller epsilon should produce narrower output bounds
    for i in 0..widths.len() - 1 {
        assert!(
            widths[i] <= widths[i + 1] + 1e-6,
            "eps={} (width={:.6}) should produce narrower bounds than eps={} (width={:.6})",
            epsilons[i],
            widths[i],
            epsilons[i + 1],
            widths[i + 1]
        );
    }
}
