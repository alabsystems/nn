// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! IBP compose verification tests for the Qwen3 MLP (feedforward) blocks.
//!
//! Decomposes the Qwen3 SwiGLU MLP pipeline into focused sub-graphs for
//! NY bounds propagation:
//!
//! 1. **Gate projection bounds**: Linear [S, D] -> [S, FFN_DIM]
//! 2. **Up projection bounds**: Linear [S, D] -> [S, FFN_DIM]
//! 3. **SiLU activation bounds**: element-wise SiLU preserves finiteness
//! 4. **SiluMul (SwiGLU) bounds**: SiLU(gate) * up stays bounded
//! 5. **Down projection bounds**: Linear [S, FFN_DIM] -> [S, D]
//! 6. **Full MLP block bounds**: gate+up -> SiluMul -> down
//! 7. **MLP with RMSNorm pre-normalization**
//! 8. **MoE gating**: router logits -> softmax -> top_k selection bounds
//!
//! Uses IbpValidated soundness mode per nn engineering rules.
//! Dimensions: D_MODEL=16, FFN_DIM=48, SEQ=4, NUM_EXPERTS=4.
//!
//! Part of #4186: Qwen3 MLP/FFN NY compose verification.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
    verify_and_assert, verify_and_assert_with_config,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{
    tensor_kernel_to_graph, NormBoundsMode, TensorParamBinding, VerificationSoundnessMode,
    VerifyConfig,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const D_MODEL: usize = 16;
const FFN_DIM: usize = 48;
const SEQ: usize = 4;
const NUM_EXPERTS: usize = 4;
const WEIGHT_MAG: f32 = 0.001;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

fn w(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG)
}

fn ones(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0f32)
}

// ===========================================================================
// 1. Gate projection bounds: Linear [S, D] -> [S, FFN_DIM]
// ===========================================================================

/// Build isolated gate projection sub-graph.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, FFN_DIM]`.
///
/// Qwen3 MLP gate_proj is a bias-free linear projection from hidden_dim
/// to intermediate_size. Verifies that IBP through a single linear layer
/// produces bounded output proportional to weight magnitude.
fn build_gate_projection() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_mlp_gate_proj");

    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let gate_w = b.add_input("gate_w", &[FFN_DIM, D_MODEL]);

    let out = b.add_linear(x, gate_w, None, &[SEQ, FFN_DIM]);

    b.build(out).expect("valid gate projection kernel")
}

fn gate_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
    ]
}

#[test]
fn test_qwen3_mlp_gate_proj_ibp() {
    let def = build_gate_projection();
    def.validate().expect("gate projection should validate");

    let bindings = gate_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through gate proj");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, FFN_DIM],
        "gate projection output shape should be [{SEQ}, {FFN_DIM}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 gate projection IBP: [{lo}, {hi}]");

    // With weight=WEIGHT_MAG and input in [-1, 1], output per element bounded by
    // D_MODEL * WEIGHT_MAG * 1.0 = 0.016.
    let expected_max = (D_MODEL as f32) * WEIGHT_MAG;
    assert!(
        lo >= -(expected_max + 0.01),
        "gate proj lower >= {}, got {lo}",
        -(expected_max + 0.01)
    );
    assert!(
        hi <= expected_max + 0.01,
        "gate proj upper <= {}, got {hi}",
        expected_max + 0.01
    );
}

#[test]
fn test_qwen3_mlp_gate_proj_verify_record() {
    let def = build_gate_projection();
    let bindings = gate_projection_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_mlp_gate_proj");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, FFN_DIM]
    );
}

// ===========================================================================
// 2. Up projection bounds: Linear [S, D] -> [S, FFN_DIM]
// ===========================================================================

/// Build isolated up projection sub-graph.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, FFN_DIM]`.
///
/// Structurally identical to gate_proj but verified independently since the
/// up path feeds into the multiplicative gating (not the SiLU path).
fn build_up_projection() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_mlp_up_proj");

    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let up_w = b.add_input("up_w", &[FFN_DIM, D_MODEL]);

    let out = b.add_linear(x, up_w, None, &[SEQ, FFN_DIM]);

    b.build(out).expect("valid up projection kernel")
}

fn up_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
    ]
}

#[test]
fn test_qwen3_mlp_up_proj_ibp() {
    let def = build_up_projection();
    def.validate().expect("up projection should validate");

    let bindings = up_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through up proj");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, FFN_DIM],
        "up projection output shape should be [{SEQ}, {FFN_DIM}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 up projection IBP: [{lo}, {hi}]");

    let expected_max = (D_MODEL as f32) * WEIGHT_MAG;
    assert!(
        lo >= -(expected_max + 0.01),
        "up proj lower >= {}, got {lo}",
        -(expected_max + 0.01)
    );
    assert!(
        hi <= expected_max + 0.01,
        "up proj upper <= {}, got {hi}",
        expected_max + 0.01
    );
}

#[test]
fn test_qwen3_mlp_up_proj_verify_record() {
    let def = build_up_projection();
    let bindings = up_projection_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_mlp_up_proj");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, FFN_DIM]
    );
}

// ===========================================================================
// 3. SiLU activation bounds: element-wise SiLU preserves finiteness
// ===========================================================================

/// Build isolated SiLU activation sub-graph.
///
/// Input: `[SEQ, FFN_DIM]` (Variable, representing gate_proj output).
/// Output: `[SEQ, FFN_DIM]`.
///
/// SiLU(x) = x * sigmoid(x).
/// Key property: SiLU is bounded below by ~-0.278 and monotonically
/// increasing for x > 0. IBP through sigmoid + multiply should maintain
/// finite bounds.
fn build_silu_activation() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_mlp_silu");

    let shape = [SEQ, FFN_DIM];

    let x = b.add_input("x", &shape);

    // SiLU(x) = x * sigmoid(x)
    let sig = b.add_sigmoid(x, &shape);
    let silu = b.add_binary_mul(x, sig, &shape);

    b.build(silu).expect("valid SiLU activation kernel")
}

fn silu_activation_bindings() -> Vec<TensorParamBinding> {
    vec![TensorParamBinding::Variable]
}

#[test]
fn test_qwen3_mlp_silu_ibp() {
    let def = build_silu_activation();
    def.validate().expect("SiLU activation should validate");

    let bindings = silu_activation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ, FFN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through SiLU");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, FFN_DIM],
        "SiLU output shape should be [{SEQ}, {FFN_DIM}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 SiLU activation IBP: [{lo}, {hi}]");

    // SiLU(x) for x in [-1, 1]: minimum is ~-0.278, maximum is ~0.731
    // IBP may overshoot due to interval arithmetic on x * sigmoid(x)
    assert!(lo >= -2.0, "SiLU lower should be >= -2.0, got {lo}");
    assert!(hi <= 2.0, "SiLU upper should be <= 2.0, got {hi}");
}

#[test]
fn test_qwen3_mlp_silu_crown() {
    let def = build_silu_activation();
    let bindings = silu_activation_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, FFN_DIM], 1.0);

    // SiLU has one non-linearity (sigmoid), so CROWN should produce
    // tighter bounds than IBP.
    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, FFN_DIM]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 SiLU: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_qwen3_mlp_silu_verify_record() {
    let def = build_silu_activation();
    let bindings = silu_activation_bindings();
    let input = uniform_bounds(&[SEQ, FFN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_mlp_silu_activation");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, FFN_DIM]
    );
}

// ===========================================================================
// 4. SiluMul (SwiGLU) bounds: SiLU(gate) * up stays bounded
// ===========================================================================

/// Build SiluMul (SwiGLU gating) sub-graph.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, FFN_DIM]`.
///
/// Composes: gate_proj -> SiLU -> mul(up_proj).
/// This is the core SwiGLU computation without the down projection.
///
/// Key property: the multiplicative interaction between SiLU(gate) and up
/// could amplify bounds, but with small weights the output stays bounded.
fn build_silumul_gating() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_mlp_silumul");

    let shape = [SEQ, D_MODEL];
    let ffn_shape = [SEQ, FFN_DIM];

    let x = b.add_input("x", &shape);
    let gate_w = b.add_input("gate_w", &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input("up_w", &[FFN_DIM, D_MODEL]);

    // gate_proj(x)
    let gate_proj = b.add_linear(x, gate_w, None, &ffn_shape);
    // SiLU(gate_proj(x))
    let gate_sig = b.add_sigmoid(gate_proj, &ffn_shape);
    let gate_act = b.add_binary_mul(gate_proj, gate_sig, &ffn_shape);
    // up_proj(x)
    let up_proj = b.add_linear(x, up_w, None, &ffn_shape);
    // SiLU(gate) * up
    let out = b.add_binary_mul(gate_act, up_proj, &ffn_shape);

    b.build(out).expect("valid SiluMul gating kernel")
}

fn silumul_gating_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
    ]
}

#[test]
fn test_qwen3_mlp_silumul_ibp() {
    let def = build_silumul_gating();
    def.validate().expect("SiluMul gating should validate");

    let bindings = silumul_gating_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Only `x` is a Variable (NETWORK_INPUT sentinel); the 2 weights fold into
    // their linears. Translated ops: gate linear (1) + sigmoid (1) + silu mul (1)
    // + up linear (1) + gated mul (1) = 5 nodes.
    assert!(
        graph.num_nodes() >= 5,
        "SiluMul graph >= 5 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through SiluMul");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, FFN_DIM],
        "SiluMul output shape should be [{SEQ}, {FFN_DIM}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 SiluMul (SwiGLU gating) IBP: [{lo}, {hi}]");

    // With small weights (WEIGHT_MAG=0.001) and input [-1, 1]:
    // gate_proj output ~ [-0.016, 0.016], SiLU output ~ [-0.005, 0.012]
    // up_proj output ~ [-0.016, 0.016]
    // Product: magnitude < 0.001. IBP may widen.
    assert!(lo.abs() < 1e3, "SiluMul lower magnitude < 1e3, got {lo}");
    assert!(hi.abs() < 1e3, "SiluMul upper magnitude < 1e3, got {hi}");
}

#[test]
fn test_qwen3_mlp_silumul_verify_record() {
    let def = build_silumul_gating();
    let bindings = silumul_gating_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_mlp_silumul_gating");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, FFN_DIM]
    );
}

// ===========================================================================
// 5. Down projection bounds: Linear [S, FFN_DIM] -> [S, D]
// ===========================================================================

/// Build isolated down projection sub-graph.
///
/// Input: `[SEQ, FFN_DIM]` (Variable, representing SwiGLU gated output).
/// Output: `[SEQ, D_MODEL]`.
///
/// The down projection maps from intermediate_size back to hidden_dim.
fn build_down_projection() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_mlp_down_proj");

    let x = b.add_input("x", &[SEQ, FFN_DIM]);
    let down_w = b.add_input("down_w", &[D_MODEL, FFN_DIM]);

    let out = b.add_linear(x, down_w, None, &[SEQ, D_MODEL]);

    b.build(out).expect("valid down projection kernel")
}

fn down_projection_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])),
    ]
}

#[test]
fn test_qwen3_mlp_down_proj_ibp() {
    let def = build_down_projection();
    def.validate().expect("down projection should validate");

    let bindings = down_projection_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ, FFN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through down proj");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, D_MODEL],
        "down projection output shape should be [{SEQ}, {D_MODEL}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 down projection IBP: [{lo}, {hi}]");

    // With weight=WEIGHT_MAG and input in [-1, 1], output per element bounded by
    // FFN_DIM * WEIGHT_MAG * 1.0 = 0.048.
    let expected_max = (FFN_DIM as f32) * WEIGHT_MAG;
    assert!(
        lo >= -(expected_max + 0.01),
        "down proj lower >= {}, got {lo}",
        -(expected_max + 0.01)
    );
    assert!(
        hi <= expected_max + 0.01,
        "down proj upper <= {}, got {hi}",
        expected_max + 0.01
    );
}

#[test]
fn test_qwen3_mlp_down_proj_verify_record() {
    let def = build_down_projection();
    let bindings = down_projection_bindings();
    let input = uniform_bounds(&[SEQ, FFN_DIM], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_mlp_down_proj");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, D_MODEL]
    );
}

// ===========================================================================
// 6. Full MLP block bounds: gate+up -> SiluMul -> down
// ===========================================================================

/// Build the complete Qwen3 SwiGLU MLP block.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
///
/// Full SwiGLU pipeline:
///   gate_proj(x) -> SiLU(gate) * up_proj(x) -> down_proj -> output
///
/// This composes tests 1-5 into the end-to-end MLP verification.
fn build_full_mlp_block() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_mlp_full_block");

    let shape = [SEQ, D_MODEL];
    let ffn_shape = [SEQ, FFN_DIM];

    let x = b.add_input("x", &shape);
    let gate_w = b.add_input("gate_w", &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input("up_w", &[FFN_DIM, D_MODEL]);
    let down_w = b.add_input("down_w", &[D_MODEL, FFN_DIM]);

    // SwiGLU: silu(gate_proj(x)) * up_proj(x) -> down_proj
    let gate_proj = b.add_linear(x, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate_proj, &ffn_shape);
    let gate_act = b.add_binary_mul(gate_proj, gate_sig, &ffn_shape);
    let up_proj = b.add_linear(x, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up_proj, &ffn_shape);
    let out = b.add_linear(gated, down_w, None, &shape);

    b.build(out).expect("valid full MLP block kernel")
}

fn full_mlp_block_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])),
    ]
}

#[test]
fn test_qwen3_mlp_full_block_ibp() {
    let def = build_full_mlp_block();
    def.validate().expect("full MLP block should validate");

    let bindings = full_mlp_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // Only `x` is a Variable (NETWORK_INPUT sentinel); the 3 weights fold into
    // their linears. Translated ops: gate linear (1) + sigmoid (1) + silu mul (1)
    // + up linear (1) + gated mul (1) + down linear (1) = 6 nodes.
    assert!(
        graph.num_nodes() >= 6,
        "full MLP block graph >= 6 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through full MLP block");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, D_MODEL],
        "full MLP block output shape should be [{SEQ}, {D_MODEL}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 full MLP block IBP: [{lo}, {hi}]");

    // With small weights, the full MLP output should be bounded
    assert!(
        lo.abs() < 1e6,
        "full MLP block lower magnitude < 1e6, got {lo}"
    );
    assert!(
        hi.abs() < 1e6,
        "full MLP block upper magnitude < 1e6, got {hi}"
    );
}

#[test]
fn test_qwen3_mlp_full_block_crown() {
    let def = build_full_mlp_block();
    let bindings = full_mlp_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 full MLP block: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_qwen3_mlp_full_block_verify_record() {
    let def = build_full_mlp_block();
    let bindings = full_mlp_block_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_mlp_full_block");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, D_MODEL]
    );
}

// ===========================================================================
// 7. MLP with RMSNorm pre-normalization
// ===========================================================================

/// Build RMSNorm + SwiGLU MLP + residual.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, D_MODEL]`.
///
/// Models the Qwen3 decoder's pre-MLP normalization path:
///   normed = rms_norm(x)
///   mlp_out = swiglu(normed)
///   output = x + mlp_out (residual connection)
///
/// Uses Conservative soundness mode for Sound classification through
/// the normalization layer (per nn engineering rules: IbpValidated, not Sound).
fn build_mlp_with_rmsnorm() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_mlp_rmsnorm_block");

    let shape = [SEQ, D_MODEL];
    let ffn_shape = [SEQ, FFN_DIM];

    let x = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);
    let rms_w = b.add_input("rms_w", &[D_MODEL]);
    let gate_w = b.add_input("gate_w", &[FFN_DIM, D_MODEL]);
    let up_w = b.add_input("up_w", &[FFN_DIM, D_MODEL]);
    let down_w = b.add_input("down_w", &[D_MODEL, FFN_DIM]);

    // RMSNorm
    let normed = b.add_rms_norm(x, eps, 1, rms_w, &shape);

    // SwiGLU MLP
    let gate_proj = b.add_linear(normed, gate_w, None, &ffn_shape);
    let gate_sig = b.add_sigmoid(gate_proj, &ffn_shape);
    let gate_act = b.add_binary_mul(gate_proj, gate_sig, &ffn_shape);
    let up_proj = b.add_linear(normed, up_w, None, &ffn_shape);
    let gated = b.add_binary_mul(gate_act, up_proj, &ffn_shape);
    let mlp_out = b.add_linear(gated, down_w, None, &shape);

    // Residual connection
    let out = b.add_binary_add(x, mlp_out, &shape);

    b.build(out).expect("valid MLP with RMSNorm kernel")
}

fn mlp_with_rmsnorm_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ones(&[D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[FFN_DIM, D_MODEL])),
        TensorParamBinding::ConstantTensor(w(&[D_MODEL, FFN_DIM])),
    ]
}

#[test]
fn test_qwen3_mlp_rmsnorm_ibp() {
    let def = build_mlp_with_rmsnorm();
    def.validate().expect("MLP with RMSNorm should validate");

    let bindings = mlp_with_rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // RMSNorm fuses to 1 native node; SwiGLU = gate linear (1) + sigmoid (1) +
    // silu mul (1) + up linear (1) + gated mul (1) + down linear (1) = 6; plus
    // the residual add (1). Only `x` is a Variable (NETWORK_INPUT sentinel) and
    // the 5 weights fold into their ops, so the graph is 1 + 6 + 1 = 8 nodes.
    assert!(
        graph.num_nodes() >= 8,
        "MLP with RMSNorm graph >= 8 nodes, got {}",
        graph.num_nodes()
    );

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through MLP with RMSNorm");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, D_MODEL],
        "MLP with RMSNorm output shape should be [{SEQ}, {D_MODEL}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 MLP with RMSNorm IBP: [{lo}, {hi}]");
    assert!(
        lo.abs() < 1e6,
        "MLP with RMSNorm lower magnitude < 1e6, got {lo}"
    );
    assert!(
        hi.abs() < 1e6,
        "MLP with RMSNorm upper magnitude < 1e6, got {hi}"
    );
}

#[test]
fn test_qwen3_mlp_rmsnorm_conservative_sound() {
    let def = build_mlp_with_rmsnorm();
    let bindings = mlp_with_rmsnorm_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "qwen3_mlp_rmsnorm_block",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative RMSNorm + MLP should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    assert_bounds_valid(&result.output_bounds);
    let (lo, hi) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "Qwen3 MLP with RMSNorm (Conservative): bounds=[{lo}, {hi}], soundness={:?}",
        result.verification.soundness_mode
    );
}

#[test]
fn test_qwen3_mlp_rmsnorm_crown() {
    let def = build_mlp_with_rmsnorm();
    let bindings = mlp_with_rmsnorm_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, D_MODEL]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 MLP with RMSNorm: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

// ===========================================================================
// 8. MoE gating: router logits -> softmax -> top_k selection bounds
// ===========================================================================

/// Build MoE router gating sub-graph.
///
/// Input: `[SEQ, D_MODEL]` (Variable).
/// Output: `[SEQ, NUM_EXPERTS]` (routing probabilities after softmax).
///
/// The MoE router computes:
///   logits = Linear(x, router_w)       [SEQ, NUM_EXPERTS]
///   probs  = softmax(logits, dim=1)    [SEQ, NUM_EXPERTS]
///
/// Key property: softmax output is always in [0, 1] and rows sum to 1,
/// regardless of input magnitude. This ensures routing weights are valid
/// probability distributions.
fn build_moe_router_gating() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("qwen3_mlp_moe_router");

    let x = b.add_input("x", &[SEQ, D_MODEL]);
    let router_w = b.add_input("router_w", &[NUM_EXPERTS, D_MODEL]);

    let logits = b.add_linear(x, router_w, None, &[SEQ, NUM_EXPERTS]);
    let probs = b.add_softmax(logits, 1, &[SEQ, NUM_EXPERTS]);

    b.build(probs).expect("valid MoE router gating kernel")
}

fn moe_router_gating_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(w(&[NUM_EXPERTS, D_MODEL])),
    ]
}

#[test]
fn test_qwen3_mlp_moe_router_ibp() {
    let def = build_moe_router_gating();
    def.validate().expect("MoE router gating should validate");

    let bindings = moe_router_gating_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through MoE router");
    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ, NUM_EXPERTS],
        "MoE router output shape should be [{SEQ}, {NUM_EXPERTS}]"
    );
    assert_bounds_valid(&output);

    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE router gating IBP: [{lo}, {hi}]");

    // Softmax output is always in [0, 1]. IBP may slightly overshoot.
    assert!(
        lo >= -0.01,
        "MoE router softmax lower should be >= -0.01, got {lo}"
    );
    assert!(
        hi <= 1.01,
        "MoE router softmax upper should be <= 1.01, got {hi}"
    );
}

#[test]
fn test_qwen3_mlp_moe_router_crown() {
    let def = build_moe_router_gating();
    let bindings = moe_router_gating_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    // Linear + softmax: CROWN should handle this well since it linearizes
    // around the midpoint of the softmax.
    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[SEQ, NUM_EXPERTS]);
    let (lo, hi) = bounds_min_max(&output);
    eprintln!("Qwen3 MoE router: method={method:?}, [{lo}, {hi}]");
    if let Some(r) = &fallback_reason {
        eprintln!("Fallback: {r}");
    }
}

#[test]
fn test_qwen3_mlp_moe_router_verify_record() {
    let def = build_moe_router_gating();
    let bindings = moe_router_gating_bindings();
    let input = uniform_bounds(&[SEQ, D_MODEL], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "qwen3_mlp_moe_router_gating");
    assert_eq!(result.num_variables, 1);
    assert_eq!(
        result.output_bounds.lower_upper().0.shape(),
        &[SEQ, NUM_EXPERTS]
    );
}
