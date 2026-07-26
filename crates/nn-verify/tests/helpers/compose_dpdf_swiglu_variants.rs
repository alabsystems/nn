// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for SwiGLU FFN variants: gate patterns, dimension scaling,
//! depth composition.
//!
//! Verifies IBP and CROWN bound propagation through SwiGLU feed-forward network
//! variants used across dpdf models (GLM-OCR, Qwen3-VL, Granite-Docling,
//! FireRed-OCR). SwiGLU is the dominant FFN architecture in modern LLMs:
//! `SwiGLU(x) = (gate_proj(x) * SiLU(gate_proj(x))) * up_proj(x)` followed
//! by `down_proj`.
//!
//! 1.  **Standard SwiGLU IBP + CROWN**: gate_proj -> SiLU -> mul(up_proj) -> down_proj
//! 2.  **SwiGLU dimension ratio**: 2/3 * 4h hidden dimension scaling (IBP)
//! 3.  **SwiGLU with RMSNorm**: pre-norm -> SwiGLU composition (IBP + CROWN)
//! 4.  **SwiGLU residual**: x + SwiGLU(RMSNorm(x)) (IBP)
//! 5.  **SwiGLU at different scales**: 256, 512, 1024 hidden dims (IBP)
//! 6.  **SwiGLU depth 2**: stacked FFN layers (IBP + CROWN)
//! 7.  **SwiGLU depth 4**: deep FFN chain bound widening (IBP)
//! 8.  **SwiGLU gate analysis**: sigmoid gate bounded in (0, 1) (IBP)
//! 9.  **SwiGLU vs GELU FFN**: bound width comparison (IBP)
//! 10. **SwiGLU with dropout**: stochastic depth skip (IBP)
//! 11. **Quantized SwiGLU**: INT4 gate/up projections (IBP)
//! 12. **SwiGLU monotone tightening**: smaller eps -> tighter output (IBP)
//! 13. **SwiGLU + attention**: decoder block FFN component (IBP + CROWN)
//! 14. **MoE SwiGLU**: expert FFN with gated routing (IBP)
//! 15. **SwiGLU numerical stability**: large input range handling (IBP)
//!
//! Architecture references:
//! - SwiGLU (Shazeer, 2020): SiLU-gated linear unit FFN
//! - Llama (Touvron et al., 2023): SwiGLU with 2/3 * 4h intermediate
//! - GLM-4V (THUDM): SwiGLU in decoder layers
//! - Qwen3-VL (Alibaba): SwiGLU with GQA attention
//! - Granite-Docling: SwiGLU in Granite LLM decoder
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=64, FFN_DIM=128
//!
//! Part of #4004: SwiGLU FFN variant compose tests for dpdf models.

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
const HIDDEN_DIM: usize = 64;
const FFN_DIM: usize = 128;
const WEIGHT_MAG: f32 = 0.02;
const NUM_HEADS: usize = 4;
const NUM_EXPERTS: usize = 4;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build SiLU activation as a single fused node: SiLU(x) = x * sigmoid(x).
///
/// Emitting the fused `Silu` op (instead of `sigmoid` + `binary_mul`) lets ny
/// recognize the downstream `MulBinary(SiLU(gate), up)` SwiGLU pattern and
/// apply its up/gate-correlation zonotope tightening.
fn add_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    b.add_silu(input, shape)
}

/// Build a standard SwiGLU FFN block.
///
/// Pattern: gate_proj(x) -> SiLU -> mul(up_proj(x)) -> down_proj
///
/// Input shape: `[seq_len, hidden_dim]`.
/// Output shape: `[seq_len, hidden_dim]`.
fn build_swiglu_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
    hidden_dim: usize,
    ffn_dim: usize,
) -> nn_dsl::TensorNodeId {
    let ffn_shape = [seq_len, ffn_dim];
    let out_shape = [seq_len, hidden_dim];

    let gate_w = b.add_input(&format!("{prefix}_gate_w"), &[ffn_dim, hidden_dim]);
    let up_w = b.add_input(&format!("{prefix}_up_w"), &[ffn_dim, hidden_dim]);
    let down_w = b.add_input(&format!("{prefix}_down_w"), &[hidden_dim, ffn_dim]);

    // gate_proj -> SiLU
    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(b, gate, &ffn_shape);

    // up_proj
    let up = b.add_linear(input, up_w, None, &ffn_shape);

    // element-wise gate * up -> down_proj
    let hidden = b.add_binary_mul(gate_act, up, &ffn_shape);
    b.add_linear(hidden, down_w, None, &out_shape)
}

/// Push SwiGLU weight bindings (gate_w, up_w, down_w) for given dimensions.
fn push_swiglu_bindings(
    bindings: &mut Vec<TensorParamBinding>,
    hidden_dim: usize,
    ffn_dim: usize,
    weight_mag: f32,
) {
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ffn_dim, hidden_dim]),
        weight_mag,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[ffn_dim, hidden_dim]),
        weight_mag,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[hidden_dim, ffn_dim]),
        weight_mag,
    )));
}

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ===========================================================================
// 1. Standard SwiGLU IBP + CROWN
// ===========================================================================

fn build_standard_swiglu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_swiglu_standard");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_swiglu_block(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    b.build(out).expect("valid standard SwiGLU kernel")
}

fn standard_swiglu_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

#[test]
fn test_swiglu_standard_ibp() {
    let def = build_standard_swiglu_kernel();
    let bindings = standard_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SwiGLU standard IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_swiglu_standard_crown() {
    let def = build_standard_swiglu_kernel();
    let bindings = standard_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SwiGLU standard CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 2. SwiGLU dimension ratio: 2/3 * 4h hidden dimension scaling (IBP)
// ===========================================================================

#[test]
fn test_swiglu_dimension_ratio_ibp() {
    // Llama-style: intermediate = 2/3 * 4 * hidden = 8/3 * hidden
    // For HIDDEN_DIM=64: ffn_dim = 8/3 * 64 ≈ 170, round to 168 (divisible by 8)
    let hidden = 64;
    let ffn_scaled = 168; // ≈ 2/3 * 4 * 64

    let mut b = TensorBlockBuilder::new("dpdf_swiglu_dim_ratio");
    let input = b.add_input("x", &[SEQ_LEN, hidden]);
    let out = build_swiglu_block(&mut b, input, "ffn", SEQ_LEN, hidden, ffn_scaled);
    let def = b.build(out).expect("valid dimension ratio kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings, hidden, ffn_scaled, WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, hidden], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("SwiGLU dim ratio (hidden={hidden}, ffn={ffn_scaled}) IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

// ===========================================================================
// 3. SwiGLU with RMSNorm: pre-norm -> SwiGLU composition (IBP + CROWN)
// ===========================================================================

fn build_rmsnorm_swiglu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_swiglu_rmsnorm");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let norm_weight = b.add_input("norm_weight", &[HIDDEN_DIM]);

    // RMSNorm -> SwiGLU
    let normed = b.add_rms_norm(input, eps, 1, norm_weight, &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_swiglu_block(&mut b, normed, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    b.build(out).expect("valid RMSNorm + SwiGLU kernel")
}

fn rmsnorm_swiglu_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
    ];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

#[test]
fn test_swiglu_rmsnorm_ibp() {
    let def = build_rmsnorm_swiglu_kernel();
    let bindings = rmsnorm_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("RMSNorm + SwiGLU IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

#[test]
fn test_swiglu_rmsnorm_crown() {
    let def = build_rmsnorm_swiglu_kernel();
    let bindings = rmsnorm_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("RMSNorm + SwiGLU CROWN: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 4. SwiGLU residual: x + SwiGLU(RMSNorm(x)) (IBP)
// ===========================================================================

fn build_swiglu_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_swiglu_residual");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let norm_weight = b.add_input("norm_weight", &[HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Pre-norm residual: x + SwiGLU(RMSNorm(x))
    let normed = b.add_rms_norm(input, eps, 1, norm_weight, &shape);
    let ffn_out = build_swiglu_block(&mut b, normed, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let out = b.add_binary_add(input, ffn_out, &shape);

    b.build(out).expect("valid SwiGLU residual kernel")
}

fn swiglu_residual_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
    ];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

#[test]
fn test_swiglu_residual_ibp() {
    let def = build_swiglu_residual_kernel();
    let bindings = swiglu_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Pre-norm residual x + SwiGLU(RMSNorm(x)). The RMSNorm output z satisfies the
    // exact joint bound ‖z‖₂ ≤ √n that box-IBP/CROWN otherwise see only as the
    // decorrelated per-coordinate box (|z_i| ≤ √n). ny now carries that sphere as
    // an L2 constraint on the RMSNorm IBP output and the downstream gate/up Linear
    // intersects its box interval with the EXACT Cauchy–Schwarz row bound
    // (‖w‖₂·√n instead of ‖w‖₁·√n). That alone collapses the residual lower from
    // ~-269 to ~-4.3 — a plain, sound IBP pass now clears the >-100 target with no
    // beta-CROWN search required. Intersection only tightens, so the bound remains
    // a sound enclosure; the threshold is NOT weakened.
    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SwiGLU residual IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(
        lo_min > -100.0,
        "RMSNorm L2 (Cauchy–Schwarz) tightening must SOUNDLY pull the residual \
         lower above -100 via plain IBP, got {lo_min}"
    );
}

// ===========================================================================
// 5. SwiGLU at different scales: 256, 512, 1024 hidden dims (IBP)
// ===========================================================================

fn test_swiglu_at_scale(hidden_dim: usize) {
    let ffn_dim = hidden_dim * 2; // Standard 2x expansion
    let mut b = TensorBlockBuilder::new(&format!("dpdf_swiglu_scale_{hidden_dim}"));
    let input = b.add_input("x", &[SEQ_LEN, hidden_dim]);
    let out = build_swiglu_block(&mut b, input, "ffn", SEQ_LEN, hidden_dim, ffn_dim);
    let def = b.build(out).expect("valid scale kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings, hidden_dim, ffn_dim, WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, hidden_dim], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("SwiGLU scale hidden_dim={hidden_dim} IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

#[test]
fn test_swiglu_scale_256() {
    test_swiglu_at_scale(256);
}

#[test]
fn test_swiglu_scale_512() {
    test_swiglu_at_scale(512);
}

#[test]
fn test_swiglu_scale_1024() {
    test_swiglu_at_scale(1024);
}

// ===========================================================================
// 6. SwiGLU depth 2: stacked FFN layers (IBP + CROWN)
// ===========================================================================

fn build_swiglu_depth2_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_swiglu_depth2");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    let h = build_swiglu_block(&mut b, input, "ffn0", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let out = build_swiglu_block(&mut b, h, "ffn1", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    b.build(out).expect("valid SwiGLU depth-2 kernel")
}

fn swiglu_depth2_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

#[test]
fn test_swiglu_depth2_ibp() {
    let def = build_swiglu_depth2_kernel();
    let bindings = swiglu_depth2_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("SwiGLU depth-2 IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

#[test]
fn test_swiglu_depth2_crown() {
    let def = build_swiglu_depth2_kernel();
    let bindings = swiglu_depth2_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("SwiGLU depth-2 CROWN: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 7. SwiGLU depth 4: deep FFN chain bound widening (IBP)
// ===========================================================================

#[test]
fn test_swiglu_depth4_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_swiglu_depth4");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    let h = build_swiglu_block(&mut b, input, "ffn0", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let h = build_swiglu_block(&mut b, h, "ffn1", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let h = build_swiglu_block(&mut b, h, "ffn2", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let out = build_swiglu_block(&mut b, h, "ffn3", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let def = b.build(out).expect("valid SwiGLU depth-4 kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..4 {
        push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("SwiGLU depth-4 IBP: width={width:.6}");
    assert!(width.is_finite(), "depth-4 output width must be finite");
}

// ===========================================================================
// 8. SwiGLU gate analysis: sigmoid gate bounded in (0, 1) (IBP)
// ===========================================================================

/// Build the gate path only: gate_proj(x) -> sigmoid.
/// This isolates the gating mechanism to verify sigmoid output in (0, 1).
fn build_swiglu_gate_only_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_swiglu_gate_only");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let gate_w = b.add_input("gate_w", &[FFN_DIM, HIDDEN_DIM]);

    let gate = b.add_linear(input, gate_w, None, &[SEQ_LEN, FFN_DIM]);
    let out = b.add_sigmoid(gate, &[SEQ_LEN, FFN_DIM]);

    b.build(out).expect("valid SwiGLU gate-only kernel")
}

#[test]
fn test_swiglu_gate_sigmoid_bounded_01_ibp() {
    let def = build_swiglu_gate_only_kernel();
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 2.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("SwiGLU gate sigmoid IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min >= 0.0 - tol,
        "sigmoid gate lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "sigmoid gate upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 9. SwiGLU vs GELU FFN: bound width comparison (IBP)
// ===========================================================================

fn build_gelu_ffn_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_gelu_ffn");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w1 = b.add_input("w1", &[FFN_DIM, HIDDEN_DIM]);
    let w2 = b.add_input("w2", &[HIDDEN_DIM, FFN_DIM]);

    let h = b.add_linear(input, w1, None, &[SEQ_LEN, FFN_DIM]);
    let h = b.add_gelu(h, &[SEQ_LEN, FFN_DIM]);
    let out = b.add_linear(h, w2, None, &[SEQ_LEN, HIDDEN_DIM]);

    b.build(out).expect("valid GELU FFN kernel")
}

#[test]
fn test_swiglu_vs_gelu_ffn_bound_width_ibp() {
    // SwiGLU FFN
    let swiglu_def = build_standard_swiglu_kernel();
    let swiglu_bindings = standard_swiglu_bindings();
    let swiglu_graph = tensor_kernel_to_graph(&swiglu_def, &swiglu_bindings).expect("SwiGLU graph");

    // GELU FFN
    let gelu_def = build_gelu_ffn_kernel();
    let gelu_bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[FFN_DIM, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[HIDDEN_DIM, FFN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    let gelu_graph = tensor_kernel_to_graph(&gelu_def, &gelu_bindings).expect("GELU graph");

    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let swiglu_output = swiglu_graph.propagate_ibp(&input).expect("SwiGLU IBP");
    let gelu_output = gelu_graph.propagate_ibp(&input).expect("GELU IBP");

    assert_bounds_valid(&swiglu_output);
    assert_bounds_valid(&gelu_output);

    let swiglu_width = bound_width(&swiglu_output);
    let gelu_width = bound_width(&gelu_output);
    eprintln!("SwiGLU vs GELU FFN IBP: swiglu_width={swiglu_width:.6}, gelu_width={gelu_width:.6}");
    // Both should produce finite, reasonable bounds
    assert!(swiglu_width.is_finite(), "SwiGLU width must be finite");
    assert!(gelu_width.is_finite(), "GELU width must be finite");
}

// ===========================================================================
// 10. SwiGLU with dropout: stochastic depth skip (IBP)
// ===========================================================================

/// Stochastic depth modeled as: x + alpha * SwiGLU(x), where alpha in [0, 1]
/// represents the dropout survival probability. At inference, alpha=1.
/// We verify bounds with alpha as a constant scale factor.
#[test]
fn test_swiglu_stochastic_depth_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_swiglu_stochastic_depth");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // SwiGLU path
    let ffn_out = build_swiglu_block(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Scale by survival probability (modeled as layer_scale)
    let alpha = b.add_input("alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &shape);
    let scaled = b.add_binary_mul(ffn_out, alpha_bc, &shape);

    // Residual: x + alpha * SwiGLU(x)
    let out = b.add_binary_add(input, scaled, &shape);
    let def = b.build(out).expect("valid stochastic depth kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    // alpha = 0.8 (80% survival probability)
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.8f32,
    )));

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SwiGLU stochastic depth IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 11. Quantized SwiGLU: INT4 gate/up projections (IBP)
// ===========================================================================

/// Quantized SwiGLU models INT4 dequantization as smaller weight magnitudes.
/// INT4 weights have reduced dynamic range, producing tighter bounds.
#[test]
fn test_swiglu_quantized_int4_ibp() {
    let quant_weight_mag = 0.01f32; // Tighter than FP32 WEIGHT_MAG=0.02

    let mut b = TensorBlockBuilder::new("dpdf_swiglu_quantized");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_swiglu_block(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let def = b.build(out).expect("valid quantized SwiGLU kernel");

    let mut quant_bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut quant_bindings, HIDDEN_DIM, FFN_DIM, quant_weight_mag);

    let graph = tensor_kernel_to_graph(&def, &quant_bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let quant_output = graph.propagate_ibp(&input).expect("quantized IBP");
    assert_bounds_valid(&quant_output);
    let quant_width = bound_width(&quant_output);

    // Compare with FP32 weights
    let fp32_bindings = standard_swiglu_bindings();
    let fp32_def = build_standard_swiglu_kernel();
    let fp32_graph = tensor_kernel_to_graph(&fp32_def, &fp32_bindings).expect("FP32 graph");
    let fp32_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let fp32_output = fp32_graph.propagate_ibp(&fp32_input).expect("FP32 IBP");
    assert_bounds_valid(&fp32_output);
    let fp32_width = bound_width(&fp32_output);

    eprintln!("Quantized SwiGLU IBP: quant_width={quant_width:.6}, fp32_width={fp32_width:.6}");
    // INT4 (smaller weights) should produce tighter or equal bounds
    assert!(
        quant_width <= fp32_width + 1e-4,
        "quantized bounds should be tighter: quant={quant_width}, fp32={fp32_width}"
    );
}

// ===========================================================================
// 12. SwiGLU monotone tightening: smaller eps -> tighter output (IBP)
// ===========================================================================

#[test]
fn test_swiglu_monotone_tightening_ibp() {
    let def = build_standard_swiglu_kernel();
    let bindings = standard_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let wide_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("IBP wide");
    assert_bounds_valid(&wide_output);
    let wide_width = bound_width(&wide_output);

    let tight_input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.1);
    let tight_output = graph.propagate_ibp(&tight_input).expect("IBP tight");
    assert_bounds_valid(&tight_output);
    let tight_width = bound_width(&tight_output);

    eprintln!(
        "SwiGLU monotone tightening: eps=1.0 width={wide_width:.6}, eps=0.1 width={tight_width:.6}"
    );
    assert!(
        tight_width <= wide_width + 1e-6,
        "tight input should produce tighter output: wide={wide_width}, tight={tight_width}"
    );
}

// ===========================================================================
// 13. SwiGLU + attention: decoder block FFN component (IBP + CROWN)
// ===========================================================================

/// Build a decoder block: MHA -> residual -> RMSNorm -> SwiGLU -> residual.
/// This is the standard transformer decoder pattern used in GLM-OCR, Qwen3-VL.
fn build_decoder_block_swiglu_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_swiglu_decoder_block");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Attention block: Q, K, V, Out projections
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

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

    // First residual
    let h = b.add_binary_add(input, attn_out, &shape);

    // RMSNorm before FFN
    let eps = b.add_input("ffn_norm_eps", &[1]);
    let norm_w = b.add_input("ffn_norm_w", &[HIDDEN_DIM]);
    let normed = b.add_rms_norm(h, eps, 1, norm_w, &shape);

    // SwiGLU FFN
    let ffn_out = build_swiglu_block(&mut b, normed, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Second residual
    let out = b.add_binary_add(h, ffn_out, &shape);
    b.build(out).expect("valid decoder block kernel")
}

fn decoder_block_swiglu_bindings() -> Vec<TensorParamBinding> {
    let w = |shape: &[usize]| {
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
    };
    let mut bindings = vec![
        TensorParamBinding::Variable,
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // q_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // k_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // v_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // o_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)), // norm_w
    ];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

#[test]
fn test_swiglu_decoder_block_ibp() {
    let def = build_decoder_block_swiglu_kernel();
    let bindings = decoder_block_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Decoder block + SwiGLU IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_swiglu_decoder_block_crown() {
    let def = build_decoder_block_swiglu_kernel();
    let bindings = decoder_block_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Decoder block + SwiGLU CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 14. MoE SwiGLU: expert FFN with gated routing (IBP)
// ===========================================================================

/// Build MoE SwiGLU: router -> softmax -> expert SwiGLU FFN.
/// Models the bound-critical path through one expert with gate probabilities.
#[test]
fn test_moe_swiglu_expert_routing_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_swiglu_moe_routing");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let router_w = b.add_input("router_w", &[NUM_EXPERTS, HIDDEN_DIM]);

    // Router: Linear -> softmax (gate probabilities)
    let logits = b.add_linear(input, router_w, None, &[SEQ_LEN, NUM_EXPERTS]);
    let _probs = b.add_softmax(logits, 1, &[SEQ_LEN, NUM_EXPERTS]);

    // Expert SwiGLU FFN on routed tokens
    let ffn_out = build_swiglu_block(&mut b, input, "expert0", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    let def = b.build(ffn_out).expect("valid MoE SwiGLU kernel");

    let mut bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[NUM_EXPERTS, HIDDEN_DIM]),
            WEIGHT_MAG,
        )),
    ];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("MoE SwiGLU routing IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 15. SwiGLU numerical stability: large input range handling (IBP)
// ===========================================================================

/// Verify SwiGLU produces finite bounds even with a large input range.
/// The sigmoid gate saturates for large inputs, which should keep the
/// multiplicative interaction bounded.
#[test]
fn test_swiglu_large_input_range_ibp() {
    let def = build_standard_swiglu_kernel();
    let bindings = standard_swiglu_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Large input range: [-10, 10]
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 10.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("SwiGLU large input range IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(
        lo_min.is_finite(),
        "lower bound must be finite for large inputs"
    );
    assert!(
        hi_max.is_finite(),
        "upper bound must be finite for large inputs"
    );

    // Even with large inputs, sigmoid gate saturates, keeping things bounded
    let width = hi_max - lo_min;
    assert!(
        width.is_finite(),
        "bound width must be finite even for large inputs"
    );
}
