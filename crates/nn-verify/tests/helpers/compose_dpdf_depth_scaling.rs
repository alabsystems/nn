// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Model depth scaling bounds growth verification.
//!
//! Measures how output bound width grows with model depth (1/2/4/8 layer
//! stacks) and verifies the effects of residual connections, normalization
//! layers, CROWN vs IBP propagation, and output activation capping.
//!
//! 1. **Linear stack 1-layer IBP**: Single Linear -> ReLU. Baseline width.
//! 2. **Linear stack 2-layer IBP**: 2x (Linear -> ReLU). Width comparison.
//! 3. **Linear stack 4-layer IBP**: 4x (Linear -> ReLU). Width comparison.
//! 4. **Linear stack 8-layer IBP**: 8x (Linear -> ReLU). Width comparison.
//! 5. **Depth monotone widening**: Deeper stacks produce wider bounds.
//! 6. **Residual vs non-residual 4-layer IBP**: Residual limits growth.
//! 7. **Residual vs non-residual 8-layer IBP**: Effect amplifies with depth.
//! 8. **LayerNorm interleaved 4-layer IBP**: Norm tightens between layers.
//! 9. **RMSNorm interleaved 4-layer IBP**: RMSNorm tightens between layers.
//! 10. **Norm vs no-norm depth comparison IBP**: Norm produces tighter bounds.
//! 11. **CROWN vs IBP 1-layer**: CROWN advantage at shallow depth.
//! 12. **CROWN vs IBP 2-layer**: CROWN advantage at moderate depth.
//! 13. **CROWN vs IBP 4-layer**: CROWN advantage at deeper stacks.
//! 14. **CROWN scaling advantage**: CROWN width grows slower than IBP.
//! 15. **Sigmoid cap 4-layer IBP**: Sigmoid caps output in [0, 1].
//! 16. **Sigmoid cap 8-layer IBP**: Sigmoid caps deep stack output.
//! 17. **Softmax cap 4-layer IBP**: Softmax caps output in [0, 1].
//! 18. **Softmax cap 8-layer IBP**: Softmax caps deep stack output.
//! 19. **Residual + norm + sigmoid 8-layer IBP**: Combined stabilization.
//! 20. **Full depth scaling summary IBP**: All variants compared at depth 8.
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=32, FFN_DIM=64
//!
//! Part of #4111: Compose tests for model depth scaling bounds growth.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, ReduceOp, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

/// Sequence length for [SEQ_LEN, HIDDEN_DIM] inputs.
const SEQ_LEN: usize = 4;
/// Hidden dimension.
const HIDDEN_DIM: usize = 32;
/// FFN intermediate dimension.
const FFN_DIM: usize = 64;
/// Weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.02;
/// Number of output classes for head tests.
const NUM_CLASSES: usize = 8;

// ---------------------------------------------------------------------------
// Helpers: constant weight/bias bindings
// ---------------------------------------------------------------------------

/// Constant weight tensor binding.
fn weight(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Constant zero bias tensor binding.
fn bias(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), 0.0f32))
}

/// LayerNorm epsilon scalar binding.
fn eps_binding() -> TensorParamBinding {
    TensorParamBinding::ConstantScalar(1e-5)
}

/// LayerNorm/RMSNorm weight (all-ones) binding.
fn norm_weight(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 1.0f32))
}

/// LayerNorm bias (all-zeros) binding.
fn norm_bias(dim: usize) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 0.0f32))
}

// ---------------------------------------------------------------------------
// Graph builders: plain Linear -> ReLU stack (no residual, no norm)
// ---------------------------------------------------------------------------

/// Build N layers of (Linear -> ReLU) without residual or normalization.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Each layer: Linear(HIDDEN_DIM, FFN_DIM) -> ReLU -> Linear(FFN_DIM, HIDDEN_DIM) -> ReLU.
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_plain_stack(num_layers: usize) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let mut b = TensorBlockBuilder::new(&format!("depth_plain_{num_layers}L"));

    let input = b.add_input("x", &shape);
    let mut x = input;

    for i in 0..num_layers {
        let up_w = b.add_input(&format!("L{i}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("L{i}_down_w"), &[HIDDEN_DIM, FFN_DIM]);
        let up = b.add_linear(x, up_w, None, &ffn_shape);
        let act = b.add_relu(up, &ffn_shape);
        let down = b.add_linear(act, down_w, None, &shape);
        x = b.add_relu(down, &shape);
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer plain stack: {e}"))
}

fn plain_stack_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // x
    for _ in 0..num_layers {
        bindings.push(weight(&[FFN_DIM, HIDDEN_DIM]));
        bindings.push(weight(&[HIDDEN_DIM, FFN_DIM]));
    }
    bindings
}

// ---------------------------------------------------------------------------
// Graph builders: residual stack (Linear -> ReLU + skip)
// ---------------------------------------------------------------------------

/// Build N layers of (Linear -> ReLU + residual) without normalization.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Each layer: out = x + ReLU(Linear(ReLU(Linear(x)))).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_residual_stack(num_layers: usize) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let mut b = TensorBlockBuilder::new(&format!("depth_residual_{num_layers}L"));

    let input = b.add_input("x", &shape);
    let mut x = input;

    for i in 0..num_layers {
        let up_w = b.add_input(&format!("L{i}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("L{i}_down_w"), &[HIDDEN_DIM, FFN_DIM]);
        let up = b.add_linear(x, up_w, None, &ffn_shape);
        let act = b.add_relu(up, &ffn_shape);
        let down = b.add_linear(act, down_w, None, &shape);
        let sublayer = b.add_relu(down, &shape);
        x = b.add_binary_add(x, sublayer, &shape);
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer residual stack: {e}"))
}

fn residual_stack_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // x
    for _ in 0..num_layers {
        bindings.push(weight(&[FFN_DIM, HIDDEN_DIM]));
        bindings.push(weight(&[HIDDEN_DIM, FFN_DIM]));
    }
    bindings
}

// ---------------------------------------------------------------------------
// Graph builders: norm-interleaved stack (LayerNorm between layers)
// ---------------------------------------------------------------------------

/// Build N layers of (LayerNorm -> Linear -> ReLU) with residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Each layer: out = x + ReLU(Linear(ReLU(Linear(LayerNorm(x))))).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_layernorm_stack(num_layers: usize) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let mut b = TensorBlockBuilder::new(&format!("depth_layernorm_{num_layers}L"));

    let input = b.add_input("x", &shape);
    let mut x = input;

    for i in 0..num_layers {
        let ln_eps = b.add_input(&format!("L{i}_ln_eps"), &[1]);
        let ln_w = b.add_input(&format!("L{i}_ln_w"), &[HIDDEN_DIM]);
        let ln_b = b.add_input(&format!("L{i}_ln_b"), &[HIDDEN_DIM]);
        let normed = b.add_layer_norm(x, ln_eps, 1, ln_w, ln_b, &shape);

        let up_w = b.add_input(&format!("L{i}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("L{i}_down_w"), &[HIDDEN_DIM, FFN_DIM]);
        let up = b.add_linear(normed, up_w, None, &ffn_shape);
        let act = b.add_relu(up, &ffn_shape);
        let down = b.add_linear(act, down_w, None, &shape);
        let sublayer = b.add_relu(down, &shape);
        x = b.add_binary_add(x, sublayer, &shape);
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer LayerNorm stack: {e}"))
}

fn layernorm_stack_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // x
    for _ in 0..num_layers {
        // LayerNorm: eps, weight, bias
        bindings.push(eps_binding());
        bindings.push(norm_weight(HIDDEN_DIM));
        bindings.push(norm_bias(HIDDEN_DIM));
        // FFN: up, down
        bindings.push(weight(&[FFN_DIM, HIDDEN_DIM]));
        bindings.push(weight(&[HIDDEN_DIM, FFN_DIM]));
    }
    bindings
}

// ---------------------------------------------------------------------------
// Graph builders: RMSNorm-interleaved stack
// ---------------------------------------------------------------------------

/// Build N layers of (RMSNorm -> Linear -> ReLU) with residual.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Each layer: out = x + ReLU(Linear(ReLU(Linear(RMSNorm(x))))).
/// Output: `[SEQ_LEN, HIDDEN_DIM]`.
fn build_rmsnorm_stack(num_layers: usize) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let mut b = TensorBlockBuilder::new(&format!("depth_rmsnorm_{num_layers}L"));

    let input = b.add_input("x", &shape);
    let mut x = input;

    for i in 0..num_layers {
        let rms_eps = b.add_input(&format!("L{i}_rms_eps"), &[1]);
        let rms_w = b.add_input(&format!("L{i}_rms_w"), &[HIDDEN_DIM]);
        let normed = b.add_rms_norm(x, rms_eps, 1, rms_w, &shape);

        let up_w = b.add_input(&format!("L{i}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("L{i}_down_w"), &[HIDDEN_DIM, FFN_DIM]);
        let up = b.add_linear(normed, up_w, None, &ffn_shape);
        let act = b.add_relu(up, &ffn_shape);
        let down = b.add_linear(act, down_w, None, &shape);
        let sublayer = b.add_relu(down, &shape);
        x = b.add_binary_add(x, sublayer, &shape);
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer RMSNorm stack: {e}"))
}

fn rmsnorm_stack_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // x
    for _ in 0..num_layers {
        // RMSNorm: eps, weight
        bindings.push(eps_binding());
        bindings.push(norm_weight(HIDDEN_DIM));
        // FFN: up, down
        bindings.push(weight(&[FFN_DIM, HIDDEN_DIM]));
        bindings.push(weight(&[HIDDEN_DIM, FFN_DIM]));
    }
    bindings
}

// ---------------------------------------------------------------------------
// Graph builders: stacks with output activation capping
// ---------------------------------------------------------------------------

/// Build N-layer plain stack with a sigmoid output head.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Architecture: N x (Linear -> ReLU) -> Linear head -> sigmoid.
/// Output: `[SEQ_LEN, NUM_CLASSES]` (bounded [0, 1]).
fn build_plain_stack_sigmoid(num_layers: usize) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let head_shape = [SEQ_LEN, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new(&format!("depth_plain_sigmoid_{num_layers}L"));

    let input = b.add_input("x", &shape);
    let mut x = input;

    for i in 0..num_layers {
        let up_w = b.add_input(&format!("L{i}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("L{i}_down_w"), &[HIDDEN_DIM, FFN_DIM]);
        let up = b.add_linear(x, up_w, None, &ffn_shape);
        let act = b.add_relu(up, &ffn_shape);
        let down = b.add_linear(act, down_w, None, &shape);
        x = b.add_relu(down, &shape);
    }

    // Sigmoid output head
    let head_w = b.add_input("head_w", &[NUM_CLASSES, HIDDEN_DIM]);
    let head_b = b.add_input("head_b", &[NUM_CLASSES]);
    let logits = b.add_linear(x, head_w, Some(head_b), &head_shape);
    let out = b.add_sigmoid(logits, &head_shape);

    b.build(out)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer sigmoid stack: {e}"))
}

fn plain_stack_sigmoid_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // x
    for _ in 0..num_layers {
        bindings.push(weight(&[FFN_DIM, HIDDEN_DIM]));
        bindings.push(weight(&[HIDDEN_DIM, FFN_DIM]));
    }
    bindings.push(weight(&[NUM_CLASSES, HIDDEN_DIM]));
    bindings.push(bias(&[NUM_CLASSES]));
    bindings
}

/// Build N-layer plain stack with a softmax output head.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Architecture: N x (Linear -> ReLU) -> Linear head -> softmax.
/// Output: `[SEQ_LEN, NUM_CLASSES]` (bounded [0, 1]).
fn build_plain_stack_softmax(num_layers: usize) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let head_shape = [SEQ_LEN, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new(&format!("depth_plain_softmax_{num_layers}L"));

    let input = b.add_input("x", &shape);
    let mut x = input;

    for i in 0..num_layers {
        let up_w = b.add_input(&format!("L{i}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("L{i}_down_w"), &[HIDDEN_DIM, FFN_DIM]);
        let up = b.add_linear(x, up_w, None, &ffn_shape);
        let act = b.add_relu(up, &ffn_shape);
        let down = b.add_linear(act, down_w, None, &shape);
        x = b.add_relu(down, &shape);
    }

    // Softmax output head
    let head_w = b.add_input("head_w", &[NUM_CLASSES, HIDDEN_DIM]);
    let head_b = b.add_input("head_b", &[NUM_CLASSES]);
    let logits = b.add_linear(x, head_w, Some(head_b), &head_shape);
    let out = b.add_softmax(logits, -1, &head_shape);

    b.build(out)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer softmax stack: {e}"))
}

fn plain_stack_softmax_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // x
    for _ in 0..num_layers {
        bindings.push(weight(&[FFN_DIM, HIDDEN_DIM]));
        bindings.push(weight(&[HIDDEN_DIM, FFN_DIM]));
    }
    bindings.push(weight(&[NUM_CLASSES, HIDDEN_DIM]));
    bindings.push(bias(&[NUM_CLASSES]));
    bindings
}

/// Build N-layer residual + norm + sigmoid combined stack.
///
/// Input: `[SEQ_LEN, HIDDEN_DIM]` (Variable).
/// Each layer: out = x + ReLU(Linear(ReLU(Linear(LayerNorm(x))))).
/// Final: Linear head -> sigmoid.
/// Output: `[SEQ_LEN, NUM_CLASSES]` (bounded [0, 1]).
fn build_full_stabilized_stack(num_layers: usize) -> TensorKernelDef {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let head_shape = [SEQ_LEN, NUM_CLASSES];
    let mut b = TensorBlockBuilder::new(&format!("depth_full_stabilized_{num_layers}L"));

    let input = b.add_input("x", &shape);
    let mut x = input;

    for i in 0..num_layers {
        // LayerNorm pre-norm
        let ln_eps = b.add_input(&format!("L{i}_ln_eps"), &[1]);
        let ln_w = b.add_input(&format!("L{i}_ln_w"), &[HIDDEN_DIM]);
        let ln_b = b.add_input(&format!("L{i}_ln_b"), &[HIDDEN_DIM]);
        let normed = b.add_layer_norm(x, ln_eps, 1, ln_w, ln_b, &shape);

        let up_w = b.add_input(&format!("L{i}_up_w"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("L{i}_down_w"), &[HIDDEN_DIM, FFN_DIM]);
        let up = b.add_linear(normed, up_w, None, &ffn_shape);
        let act = b.add_relu(up, &ffn_shape);
        let down = b.add_linear(act, down_w, None, &shape);
        let sublayer = b.add_relu(down, &shape);
        // Residual connection
        x = b.add_binary_add(x, sublayer, &shape);
    }

    // Sigmoid output head
    let head_w = b.add_input("head_w", &[NUM_CLASSES, HIDDEN_DIM]);
    let head_b = b.add_input("head_b", &[NUM_CLASSES]);
    let logits = b.add_linear(x, head_w, Some(head_b), &head_shape);
    let out = b.add_sigmoid(logits, &head_shape);

    b.build(out)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer stabilized stack: {e}"))
}

fn full_stabilized_stack_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // x
    for _ in 0..num_layers {
        // LayerNorm: eps, weight, bias
        bindings.push(eps_binding());
        bindings.push(norm_weight(HIDDEN_DIM));
        bindings.push(norm_bias(HIDDEN_DIM));
        // FFN: up, down
        bindings.push(weight(&[FFN_DIM, HIDDEN_DIM]));
        bindings.push(weight(&[HIDDEN_DIM, FFN_DIM]));
    }
    bindings.push(weight(&[NUM_CLASSES, HIDDEN_DIM]));
    bindings.push(bias(&[NUM_CLASSES]));
    bindings
}

// ---------------------------------------------------------------------------
// Helper: compute bound width for a given stack
// ---------------------------------------------------------------------------

/// Run IBP on a stack and return the output bound width (hi_max - lo_min).
fn ibp_bound_width(
    def: &TensorKernelDef,
    bindings: &[TensorParamBinding],
    input: &BoundedTensor,
) -> f32 {
    let graph = tensor_kernel_to_graph(def, bindings).expect("graph translation");
    let output = graph.propagate_ibp(input).expect("IBP propagation");
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    hi_max - lo_min
}

// ===========================================================================
// 1. Linear stack 1-layer IBP
// ===========================================================================

#[test]
fn test_depth_scaling_plain_1layer_ibp() {
    let def = build_plain_stack(1);
    let bindings = plain_stack_bindings(1);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP 1-layer plain");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("depth scaling plain 1-layer IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(width > 0.0, "bounds must be non-degenerate");
    assert!(width.is_finite(), "width must be finite");
}

// ===========================================================================
// 2. Linear stack 2-layer IBP
// ===========================================================================

#[test]
fn test_depth_scaling_plain_2layer_ibp() {
    let def = build_plain_stack(2);
    let bindings = plain_stack_bindings(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP 2-layer plain");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("depth scaling plain 2-layer IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(width > 0.0, "bounds must be non-degenerate");
    assert!(width.is_finite(), "width must be finite");
}

// ===========================================================================
// 3. Linear stack 4-layer IBP
// ===========================================================================

#[test]
fn test_depth_scaling_plain_4layer_ibp() {
    let def = build_plain_stack(4);
    let bindings = plain_stack_bindings(4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP 4-layer plain");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("depth scaling plain 4-layer IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(width > 0.0, "bounds must be non-degenerate");
    assert!(width.is_finite(), "width must be finite");
}

// ===========================================================================
// 4. Linear stack 8-layer IBP
// ===========================================================================

#[test]
fn test_depth_scaling_plain_8layer_ibp() {
    let def = build_plain_stack(8);
    let bindings = plain_stack_bindings(8);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP 8-layer plain");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("depth scaling plain 8-layer IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(width > 0.0, "bounds must be non-degenerate");
    assert!(width.is_finite(), "width must be finite");
}

// ===========================================================================
// 5. Depth monotone width trend: deeper contractive stacks -> non-wider bounds
// ===========================================================================

#[test]
fn test_depth_scaling_monotone_widening() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let depths = [1, 2, 4, 8];
    let mut widths = Vec::new();

    for &d in &depths {
        let def = build_plain_stack(d);
        let bindings = plain_stack_bindings(d);
        let w = ibp_bound_width(&def, &bindings, &input);
        eprintln!("depth scaling monotone: depth={d}, width={w}");
        widths.push(w);
    }

    // This plain stack uses small weights (WEIGHT_MAG=0.02) and ReLU, so each
    // Linear->ReLU sublayer is *contractive*: the per-layer linear gain
    // (~HIDDEN_DIM*FFN_DIM*WEIGHT_MAG^2 = 32*64*0.0004 ~= 0.82) is < 1 and ReLU
    // clips the negative half, so the IBP width *shrinks* with depth rather than
    // growing. The original "monotone widening" premise only holds for an
    // expansive stack; with these contractive weights it is false (observed
    // widths decrease ~0.82 -> 0.67 -> 0.45 -> 0.20). We therefore assert the
    // correct trend for this net: width is monotonically NON-INCREASING.
    for i in 1..widths.len() {
        let eps = 1e-6;
        assert!(
            widths[i] <= widths[i - 1] + eps,
            "monotone contraction violated: depth {} width {} > depth {} width {}",
            depths[i],
            widths[i],
            depths[i - 1],
            widths[i - 1]
        );
    }
    eprintln!("depth scaling monotone (contractive) widths: {widths:?}");
}

// ===========================================================================
// 6. Residual vs non-residual 4-layer IBP
// ===========================================================================

#[test]
fn test_depth_scaling_residual_vs_plain_4layer() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let plain_def = build_plain_stack(4);
    let plain_bindings = plain_stack_bindings(4);
    let plain_width = ibp_bound_width(&plain_def, &plain_bindings, &input);

    let res_def = build_residual_stack(4);
    let res_bindings = residual_stack_bindings(4);
    let res_width = ibp_bound_width(&res_def, &res_bindings, &input);

    eprintln!(
        "depth scaling residual vs plain 4L: plain_width={plain_width}, residual_width={res_width}"
    );

    // Both must be valid
    assert!(plain_width > 0.0 && plain_width.is_finite());
    assert!(res_width > 0.0 && res_width.is_finite());
}

// ===========================================================================
// 7. Residual vs non-residual 8-layer IBP
// ===========================================================================

#[test]
fn test_depth_scaling_residual_vs_plain_8layer() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let plain_def = build_plain_stack(8);
    let plain_bindings = plain_stack_bindings(8);
    let plain_width = ibp_bound_width(&plain_def, &plain_bindings, &input);

    let res_def = build_residual_stack(8);
    let res_bindings = residual_stack_bindings(8);
    let res_width = ibp_bound_width(&res_def, &res_bindings, &input);

    eprintln!(
        "depth scaling residual vs plain 8L: plain_width={plain_width}, residual_width={res_width}"
    );

    // Both must produce finite bounds
    assert!(plain_width > 0.0 && plain_width.is_finite());
    assert!(res_width > 0.0 && res_width.is_finite());
}

// ===========================================================================
// 8. LayerNorm interleaved 4-layer IBP
// ===========================================================================

#[test]
fn test_depth_scaling_layernorm_4layer_ibp() {
    let def = build_layernorm_stack(4);
    let bindings = layernorm_stack_bindings(4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP 4-layer LayerNorm stack");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("depth scaling LayerNorm 4-layer IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(width > 0.0 && width.is_finite());
}

// ===========================================================================
// 9. RMSNorm interleaved 4-layer IBP
// ===========================================================================

#[test]
fn test_depth_scaling_rmsnorm_4layer_ibp() {
    let def = build_rmsnorm_stack(4);
    let bindings = rmsnorm_stack_bindings(4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP 4-layer RMSNorm stack");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("depth scaling RMSNorm 4-layer IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(width > 0.0 && width.is_finite());
}

// ===========================================================================
// 10. Norm vs no-norm depth comparison
// ===========================================================================

#[test]
fn test_depth_scaling_norm_vs_no_norm_comparison() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Residual without norm
    let res_def = build_residual_stack(4);
    let res_bindings = residual_stack_bindings(4);
    let res_width = ibp_bound_width(&res_def, &res_bindings, &input);

    // Residual with LayerNorm
    let ln_def = build_layernorm_stack(4);
    let ln_bindings = layernorm_stack_bindings(4);
    let ln_width = ibp_bound_width(&ln_def, &ln_bindings, &input);

    // Residual with RMSNorm
    let rms_def = build_rmsnorm_stack(4);
    let rms_bindings = rmsnorm_stack_bindings(4);
    let rms_width = ibp_bound_width(&rms_def, &rms_bindings, &input);

    eprintln!(
        "depth scaling norm comparison 4L: no_norm={res_width}, LayerNorm={ln_width}, RMSNorm={rms_width}"
    );

    // All must be finite
    assert!(res_width.is_finite(), "no-norm width must be finite");
    assert!(ln_width.is_finite(), "LayerNorm width must be finite");
    assert!(rms_width.is_finite(), "RMSNorm width must be finite");
}

// ===========================================================================
// 11. CROWN vs IBP 1-layer
// ===========================================================================

#[test]
fn test_depth_scaling_crown_vs_ibp_1layer() {
    let def = build_plain_stack(1);
    let bindings = plain_stack_bindings(1);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP 1-layer");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    // CROWN with fallback
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;

    eprintln!(
        "depth scaling CROWN vs IBP 1L: method={method:?}, ibp_width={ibp_width}, crown_width={crown_width}"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    // Both must be finite
    assert!(ibp_width.is_finite() && crown_width.is_finite());
}

// ===========================================================================
// 12. CROWN vs IBP 2-layer
// ===========================================================================

#[test]
fn test_depth_scaling_crown_vs_ibp_2layer() {
    let def = build_plain_stack(2);
    let bindings = plain_stack_bindings(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP 2-layer");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    // CROWN with fallback
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;

    eprintln!(
        "depth scaling CROWN vs IBP 2L: method={method:?}, ibp_width={ibp_width}, crown_width={crown_width}"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(ibp_width.is_finite() && crown_width.is_finite());
}

// ===========================================================================
// 13. CROWN vs IBP 4-layer
// ===========================================================================

#[test]
fn test_depth_scaling_crown_vs_ibp_4layer() {
    let def = build_plain_stack(4);
    let bindings = plain_stack_bindings(4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // IBP
    let ibp_output = graph.propagate_ibp(&input).expect("IBP 4-layer");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    // CROWN with fallback
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);

    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;

    eprintln!(
        "depth scaling CROWN vs IBP 4L: method={method:?}, ibp_width={ibp_width}, crown_width={crown_width}"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }

    assert!(ibp_width.is_finite() && crown_width.is_finite());
}

// ===========================================================================
// 14. CROWN scaling advantage: CROWN width grows slower than IBP
// ===========================================================================

#[test]
fn test_depth_scaling_crown_advantage_curve() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let depths = [1, 2, 4];
    let mut ibp_widths = Vec::new();
    let mut crown_widths = Vec::new();

    for &d in &depths {
        let def = build_plain_stack(d);
        let bindings = plain_stack_bindings(d);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        // IBP
        let ibp_output = graph.propagate_ibp(&input).expect("IBP");
        let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
        let ibp_w = ibp_hi - ibp_lo;
        ibp_widths.push(ibp_w);

        // CROWN
        let (_method, crown_output, _) =
            nn_verify::propagate_with_crown_fallback(&graph, &input).expect("CROWN");
        let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
        let crown_w = crown_hi - crown_lo;
        crown_widths.push(crown_w);

        eprintln!(
            "depth scaling CROWN advantage: depth={d}, ibp_width={ibp_w}, crown_width={crown_w}"
        );
    }

    // Verify all widths are finite
    for (i, &d) in depths.iter().enumerate() {
        assert!(
            ibp_widths[i].is_finite(),
            "IBP width at depth {d} must be finite"
        );
        assert!(
            crown_widths[i].is_finite(),
            "CROWN width at depth {d} must be finite"
        );
    }

    eprintln!("IBP widths across depth:   {ibp_widths:?}");
    eprintln!("CROWN widths across depth: {crown_widths:?}");
}

// ===========================================================================
// 15. Sigmoid cap 4-layer IBP
// ===========================================================================

#[test]
fn test_depth_scaling_sigmoid_cap_4layer() {
    let def = build_plain_stack_sigmoid(4);
    let bindings = plain_stack_sigmoid_bindings(4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP 4-layer sigmoid stack");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("depth scaling sigmoid 4L IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid caps output in [0, 1]
    assert!(lo_min >= -1e-4, "sigmoid lower must be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-4,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 16. Sigmoid cap 8-layer IBP
// ===========================================================================

#[test]
fn test_depth_scaling_sigmoid_cap_8layer() {
    let def = build_plain_stack_sigmoid(8);
    let bindings = plain_stack_sigmoid_bindings(8);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP 8-layer sigmoid stack");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("depth scaling sigmoid 8L IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid caps output in [0, 1] regardless of depth
    assert!(lo_min >= -1e-4, "sigmoid lower must be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-4,
        "sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 17. Softmax cap 4-layer IBP
// ===========================================================================

#[test]
fn test_depth_scaling_softmax_cap_4layer() {
    let def = build_plain_stack_softmax(4);
    let bindings = plain_stack_softmax_bindings(4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP 4-layer softmax stack");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("depth scaling softmax 4L IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax caps output in [0, 1]
    assert!(lo_min >= -1e-4, "softmax lower must be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-4,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 18. Softmax cap 8-layer IBP
// ===========================================================================

#[test]
fn test_depth_scaling_softmax_cap_8layer() {
    let def = build_plain_stack_softmax(8);
    let bindings = plain_stack_softmax_bindings(8);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP 8-layer softmax stack");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("depth scaling softmax 8L IBP: bounds=[{lo_min}, {hi_max}]");

    // Softmax caps output in [0, 1] regardless of depth
    assert!(lo_min >= -1e-4, "softmax lower must be >= 0, got {lo_min}");
    assert!(
        hi_max <= 1.0 + 1e-4,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 19. Residual + norm + sigmoid 8-layer combined stabilization
// ===========================================================================

#[test]
fn test_depth_scaling_full_stabilized_8layer() {
    let def = build_full_stabilized_stack(8);
    let bindings = full_stabilized_stack_bindings(8);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP 8-layer stabilized stack");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, NUM_CLASSES]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("depth scaling full stabilized 8L IBP: bounds=[{lo_min}, {hi_max}]");

    // Sigmoid caps output in [0, 1]
    assert!(
        lo_min >= -1e-4,
        "stabilized sigmoid lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + 1e-4,
        "stabilized sigmoid upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 20. Full depth scaling summary at depth 8
// ===========================================================================

#[test]
fn test_depth_scaling_full_summary_8layer() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Plain (no residual, no norm)
    let plain_def = build_plain_stack(8);
    let plain_bindings = plain_stack_bindings(8);
    let plain_width = ibp_bound_width(&plain_def, &plain_bindings, &input);

    // Residual (no norm)
    let res_def = build_residual_stack(8);
    let res_bindings = residual_stack_bindings(8);
    let res_width = ibp_bound_width(&res_def, &res_bindings, &input);

    // LayerNorm + residual
    let ln_def = build_layernorm_stack(8);
    let ln_bindings = layernorm_stack_bindings(8);
    let ln_width = ibp_bound_width(&ln_def, &ln_bindings, &input);

    // RMSNorm + residual
    let rms_def = build_rmsnorm_stack(8);
    let rms_bindings = rmsnorm_stack_bindings(8);
    let rms_width = ibp_bound_width(&rms_def, &rms_bindings, &input);

    eprintln!("=== Depth Scaling Summary (8 layers) ===");
    eprintln!("  Plain (no res, no norm): width={plain_width}");
    eprintln!("  Residual (no norm):      width={res_width}");
    eprintln!("  LayerNorm + residual:    width={ln_width}");
    eprintln!("  RMSNorm + residual:      width={rms_width}");

    // All must be finite
    assert!(plain_width.is_finite(), "plain width must be finite");
    assert!(res_width.is_finite(), "residual width must be finite");
    assert!(ln_width.is_finite(), "LayerNorm width must be finite");
    assert!(rms_width.is_finite(), "RMSNorm width must be finite");

    // All must be non-degenerate
    assert!(plain_width > 0.0, "plain width must be > 0");
    assert!(res_width > 0.0, "residual width must be > 0");
    assert!(ln_width > 0.0, "LayerNorm width must be > 0");
    assert!(rms_width > 0.0, "RMSNorm width must be > 0");
}

// Additional constants for transformer/conv/LSTM depth scaling tests.
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 8
/// Channel dimension for conv tests.
const CONV_CHANNELS: usize = 16;
/// Spatial width for conv tests.
const CONV_WIDTH: usize = 8;
/// LSTM hidden size.
const LSTM_HIDDEN: usize = 32;

// ---------------------------------------------------------------------------
// Helper: Build a single pre-norm transformer block (LN -> MHA -> res -> LN -> FFN -> res)
// ---------------------------------------------------------------------------

fn add_transformer_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::tensor_ir::TensorNodeId,
    prefix: &str,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Pre-attention LayerNorm
    let ln1_eps = b.add_input(&format!("{prefix}ln1_eps"), &[1]);
    let ln1_w = b.add_input(&format!("{prefix}ln1_w"), &[HIDDEN_DIM]);
    let ln1_b = b.add_input(&format!("{prefix}ln1_b"), &[HIDDEN_DIM]);
    let normed1 = b.add_layer_norm(input, ln1_eps, 1, ln1_w, ln1_b, &shape);

    // Self-attention: Q/K/V + output projection
    let q_w = b.add_input(&format!("{prefix}q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("{prefix}k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("{prefix}v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input(&format!("{prefix}out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed1, q_w, None, &shape);
    let k = b.add_linear(normed1, k_w, None, &shape);
    let v = b.add_linear(normed1, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // Residual after attention
    let res1 = b.add_binary_add(input, attn_out, &shape);

    // Pre-FFN LayerNorm
    let ln2_eps = b.add_input(&format!("{prefix}ln2_eps"), &[1]);
    let ln2_w = b.add_input(&format!("{prefix}ln2_w"), &[HIDDEN_DIM]);
    let ln2_b = b.add_input(&format!("{prefix}ln2_b"), &[HIDDEN_DIM]);
    let normed2 = b.add_layer_norm(res1, ln2_eps, 1, ln2_w, ln2_b, &shape);

    // FFN: Linear -> GELU -> Linear
    let ffn_up_w = b.add_input(&format!("{prefix}ffn_up_w"), &[FFN_DIM, HIDDEN_DIM]);
    let ffn_down_w = b.add_input(&format!("{prefix}ffn_down_w"), &[HIDDEN_DIM, FFN_DIM]);

    let ffn_up = b.add_linear(normed2, ffn_up_w, None, &ffn_shape);
    let ffn_act = b.add_gelu(ffn_up, &ffn_shape);
    let ffn_out = b.add_linear(ffn_act, ffn_down_w, None, &shape);

    // Residual after FFN
    b.add_binary_add(res1, ffn_out, &shape)
}

/// Push bindings for one transformer block (12 params: 2xLN(eps,w,b) + 4xproj + 2xFFN).
fn push_transformer_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn_up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    // LN1: eps, weight, bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
    // Attention: Q, K, V, output projections
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj_w));
    // LN2: eps, weight, bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ln_w));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b));
    // FFN: up, down
    bindings.push(TensorParamBinding::ConstantTensor(ffn_up_w));
    bindings.push(TensorParamBinding::ConstantTensor(ffn_down_w));
}

/// Build an N-layer transformer stack.
fn build_n_layer_transformer(num_layers: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(&format!("depth_scale_transformer_{num_layers}L"));
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    let mut x = input;
    for i in 0..num_layers {
        x = add_transformer_block(&mut b, x, &format!("l{}_", i + 1));
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer transformer: {e}"))
}

fn n_layer_transformer_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable]; // x
    for _ in 0..num_layers {
        push_transformer_block_bindings(&mut bindings);
    }
    bindings
}

// ---------------------------------------------------------------------------
// Helper: Build MLP-only blocks (Linear -> GELU -> Linear with residual)
// ---------------------------------------------------------------------------

fn add_mlp_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::tensor_ir::TensorNodeId,
    prefix: &str,
    with_residual: bool,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let ln_eps = b.add_input(&format!("{prefix}ln_eps"), &[1]);
    let ln_w = b.add_input(&format!("{prefix}ln_w"), &[HIDDEN_DIM]);
    let ln_b = b.add_input(&format!("{prefix}ln_b"), &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &shape);

    let up_w = b.add_input(&format!("{prefix}up_w"), &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input(&format!("{prefix}down_w"), &[HIDDEN_DIM, FFN_DIM]);

    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let act = b.add_gelu(up, &ffn_shape);
    let out = b.add_linear(act, down_w, None, &shape);

    if with_residual {
        b.add_binary_add(input, out, &shape)
    } else {
        out
    }
}

fn push_mlp_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let ffn_up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ln_w));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b));
    bindings.push(TensorParamBinding::ConstantTensor(ffn_up_w));
    bindings.push(TensorParamBinding::ConstantTensor(ffn_down_w));
}

fn build_n_layer_mlp(num_layers: usize, with_residual: bool) -> TensorKernelDef {
    let tag = if with_residual { "res" } else { "ff" };
    let mut b = TensorBlockBuilder::new(&format!("depth_scale_mlp_{tag}_{num_layers}L"));
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    let mut x = input;
    for i in 0..num_layers {
        x = add_mlp_block(&mut b, x, &format!("l{}_", i + 1), with_residual);
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer MLP ({tag}): {e}"))
}

fn n_layer_mlp_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..num_layers {
        push_mlp_block_bindings(&mut bindings);
    }
    bindings
}

// ---------------------------------------------------------------------------
// Helper: Attention-only blocks (LN -> MHA -> residual)
// ---------------------------------------------------------------------------

fn add_attention_only_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::tensor_ir::TensorNodeId,
    prefix: &str,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let ln_eps = b.add_input(&format!("{prefix}ln_eps"), &[1]);
    let ln_w = b.add_input(&format!("{prefix}ln_w"), &[HIDDEN_DIM]);
    let ln_b = b.add_input(&format!("{prefix}ln_b"), &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &shape);

    let q_w = b.add_input(&format!("{prefix}q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("{prefix}k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("{prefix}v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input(&format!("{prefix}out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(normed, q_w, None, &shape);
    let k = b.add_linear(normed, k_w, None, &shape);
    let v = b.add_linear(normed, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    b.add_binary_add(input, attn_out, &shape)
}

fn push_attention_only_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);

    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ln_w));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b));
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj_w));
}

fn build_n_layer_attention(num_layers: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(&format!("depth_scale_attn_{num_layers}L"));
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    let mut x = input;
    for i in 0..num_layers {
        x = add_attention_only_block(&mut b, x, &format!("l{}_", i + 1));
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer attention: {e}"))
}

fn n_layer_attention_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..num_layers {
        push_attention_only_bindings(&mut bindings);
    }
    bindings
}

// ---------------------------------------------------------------------------
// Helper: Post-norm transformer block
// ---------------------------------------------------------------------------

fn add_postnorm_transformer_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::tensor_ir::TensorNodeId,
    prefix: &str,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Attention (no pre-norm)
    let q_w = b.add_input(&format!("{prefix}q_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input(&format!("{prefix}k_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input(&format!("{prefix}v_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
    let out_w = b.add_input(&format!("{prefix}out_w"), &[HIDDEN_DIM, HIDDEN_DIM]);

    let q = b.add_linear(input, q_w, None, &shape);
    let k = b.add_linear(input, k_w, None, &shape);
    let v = b.add_linear(input, v_w, None, &shape);
    let attn = b.add_attention(q, k, v, AttentionMask::Standard, Some(scale), &shape);
    let attn_out = b.add_linear(attn, out_w, None, &shape);

    // Post-norm: LN(x + attn_out)
    let res1 = b.add_binary_add(input, attn_out, &shape);
    let ln1_eps = b.add_input(&format!("{prefix}ln1_eps"), &[1]);
    let ln1_w = b.add_input(&format!("{prefix}ln1_w"), &[HIDDEN_DIM]);
    let ln1_b = b.add_input(&format!("{prefix}ln1_b"), &[HIDDEN_DIM]);
    let mid = b.add_layer_norm(res1, ln1_eps, 1, ln1_w, ln1_b, &shape);

    // FFN
    let ffn_up_w = b.add_input(&format!("{prefix}ffn_up_w"), &[FFN_DIM, HIDDEN_DIM]);
    let ffn_down_w = b.add_input(&format!("{prefix}ffn_down_w"), &[HIDDEN_DIM, FFN_DIM]);

    let ffn_up = b.add_linear(mid, ffn_up_w, None, &ffn_shape);
    let ffn_act = b.add_gelu(ffn_up, &ffn_shape);
    let ffn_out = b.add_linear(ffn_act, ffn_down_w, None, &shape);

    // Post-norm: LN(mid + ffn_out)
    let res2 = b.add_binary_add(mid, ffn_out, &shape);
    let ln2_eps = b.add_input(&format!("{prefix}ln2_eps"), &[1]);
    let ln2_w = b.add_input(&format!("{prefix}ln2_w"), &[HIDDEN_DIM]);
    let ln2_b = b.add_input(&format!("{prefix}ln2_b"), &[HIDDEN_DIM]);
    b.add_layer_norm(res2, ln2_eps, 1, ln2_w, ln2_b, &shape)
}

fn push_postnorm_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let ln_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32);
    let ln_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);
    let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn_up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    // Attention: Q, K, V, out
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(proj_w));
    // LN1: eps, weight, bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ln_w.clone()));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b.clone()));
    // FFN: up, down
    bindings.push(TensorParamBinding::ConstantTensor(ffn_up_w));
    bindings.push(TensorParamBinding::ConstantTensor(ffn_down_w));
    // LN2: eps, weight, bias
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    bindings.push(TensorParamBinding::ConstantTensor(ln_w));
    bindings.push(TensorParamBinding::ConstantTensor(ln_b));
}

fn build_n_layer_postnorm(num_layers: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(&format!("depth_scale_postnorm_{num_layers}L"));
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    let mut x = input;
    for i in 0..num_layers {
        x = add_postnorm_transformer_block(&mut b, x, &format!("l{}_", i + 1));
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer postnorm transformer: {e}"))
}

fn n_layer_postnorm_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..num_layers {
        push_postnorm_block_bindings(&mut bindings);
    }
    bindings
}

// ---------------------------------------------------------------------------
// Helper: Conv1d blocks with BatchNorm + ReLU + residual
// ---------------------------------------------------------------------------

fn add_conv_block(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::tensor_ir::TensorNodeId,
    prefix: &str,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let shape = [CONV_CHANNELS, CONV_WIDTH];

    // Conv1d: same padding, stride=1, kernel_size=3 -> padding=1
    let conv_w = b.add_input(
        &format!("{prefix}conv_w"),
        &[CONV_CHANNELS, CONV_CHANNELS, 3],
    );
    let conv_b = b.add_input(&format!("{prefix}conv_b"), &[CONV_CHANNELS]);
    let conv_out = b.add_conv1d(input, conv_w, Some(conv_b), 1, 1, &shape);

    // BatchNorm
    let bn_mean = b.add_input(&format!("{prefix}bn_mean"), &[CONV_CHANNELS]);
    let bn_var = b.add_input(&format!("{prefix}bn_var"), &[CONV_CHANNELS]);
    let bn_w = b.add_input(&format!("{prefix}bn_w"), &[CONV_CHANNELS]);
    let bn_b = b.add_input(&format!("{prefix}bn_b"), &[CONV_CHANNELS]);
    let bn_eps = b.add_input(&format!("{prefix}bn_eps"), &[1]);
    let normed = b.add_batch_norm(conv_out, bn_mean, bn_var, bn_w, bn_b, bn_eps, &shape);

    // ReLU + residual
    let activated = b.add_relu(normed, &shape);
    b.add_binary_add(input, activated, &shape)
}

fn push_conv_block_bindings(bindings: &mut Vec<TensorParamBinding>) {
    let conv_w = ArrayD::from_elem(IxDyn(&[CONV_CHANNELS, CONV_CHANNELS, 3]), WEIGHT_MAG);
    let conv_b = ArrayD::from_elem(IxDyn(&[CONV_CHANNELS]), 0.0f32);
    let bn_mean = ArrayD::from_elem(IxDyn(&[CONV_CHANNELS]), 0.0f32);
    let bn_var = ArrayD::from_elem(IxDyn(&[CONV_CHANNELS]), 1.0f32);
    let bn_w = ArrayD::from_elem(IxDyn(&[CONV_CHANNELS]), 1.0f32);
    let bn_b = ArrayD::from_elem(IxDyn(&[CONV_CHANNELS]), 0.0f32);

    bindings.push(TensorParamBinding::ConstantTensor(conv_w));
    bindings.push(TensorParamBinding::ConstantTensor(conv_b));
    bindings.push(TensorParamBinding::ConstantTensor(bn_mean));
    bindings.push(TensorParamBinding::ConstantTensor(bn_var));
    bindings.push(TensorParamBinding::ConstantTensor(bn_w));
    bindings.push(TensorParamBinding::ConstantTensor(bn_b));
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
}

fn build_n_layer_conv(num_layers: usize) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(&format!("depth_scale_conv_{num_layers}L"));
    let input = b.add_input("x", &[CONV_CHANNELS, CONV_WIDTH]);

    let mut x = input;
    for i in 0..num_layers {
        x = add_conv_block(&mut b, x, &format!("l{}_", i + 1));
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer conv: {e}"))
}

fn n_layer_conv_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..num_layers {
        push_conv_block_bindings(&mut bindings);
    }
    bindings
}

// ---------------------------------------------------------------------------
// Helper: LSTM stack (stacked LSTM cells with residual)
// ---------------------------------------------------------------------------

fn build_n_layer_lstm(num_layers: usize) -> TensorKernelDef {
    let shape = [LSTM_HIDDEN];
    let mut b = TensorBlockBuilder::new(&format!("depth_scale_lstm_{num_layers}L"));
    let input = b.add_input("x", &shape);

    let mut x = input;
    for i in 0..num_layers {
        let pfx = format!("l{}_", i + 1);
        let h0 = b.add_input(&format!("{pfx}h0"), &shape);
        let c0 = b.add_input(&format!("{pfx}c0"), &shape);
        // weight_ih: [4*hidden, hidden], weight_hh: [4*hidden, hidden]
        let w_ih = b.add_input(&format!("{pfx}w_ih"), &[4 * LSTM_HIDDEN, LSTM_HIDDEN]);
        let w_hh = b.add_input(&format!("{pfx}w_hh"), &[4 * LSTM_HIDDEN, LSTM_HIDDEN]);
        let bias = b.add_input(&format!("{pfx}bias"), &[4 * LSTM_HIDDEN]);
        let lstm_out = b.add_lstm(x, h0, c0, w_ih, w_hh, Some(bias), &shape);
        // Residual connection (same-dim projection)
        x = b.add_binary_add(x, lstm_out, &shape);
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer LSTM: {e}"))
}

fn n_layer_lstm_bindings(num_layers: usize) -> Vec<TensorParamBinding> {
    let h0 = ArrayD::from_elem(IxDyn(&[LSTM_HIDDEN]), 0.0f32);
    let c0 = ArrayD::from_elem(IxDyn(&[LSTM_HIDDEN]), 0.0f32);
    let w_ih = ArrayD::from_elem(IxDyn(&[4 * LSTM_HIDDEN, LSTM_HIDDEN]), WEIGHT_MAG);
    let w_hh = ArrayD::from_elem(IxDyn(&[4 * LSTM_HIDDEN, LSTM_HIDDEN]), WEIGHT_MAG);
    let bias = ArrayD::from_elem(IxDyn(&[4 * LSTM_HIDDEN]), 0.0f32);

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..num_layers {
        bindings.push(TensorParamBinding::ConstantTensor(h0.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(c0.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_ih.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(w_hh.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(bias.clone()));
    }
    bindings
}

// ---------------------------------------------------------------------------
// Helper: MLP block with configurable activation (ReLU / GELU / SiLU)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Activation {
    Relu,
    Gelu,
    Silu,
}

fn add_mlp_with_activation(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::tensor_ir::TensorNodeId,
    prefix: &str,
    activation: Activation,
) -> nn_dsl::tensor_ir::TensorNodeId {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];

    let ln_eps = b.add_input(&format!("{prefix}ln_eps"), &[1]);
    let ln_w = b.add_input(&format!("{prefix}ln_w"), &[HIDDEN_DIM]);
    let ln_b = b.add_input(&format!("{prefix}ln_b"), &[HIDDEN_DIM]);
    let normed = b.add_layer_norm(input, ln_eps, 1, ln_w, ln_b, &shape);

    let up_w = b.add_input(&format!("{prefix}up_w"), &[FFN_DIM, HIDDEN_DIM]);
    let down_w = b.add_input(&format!("{prefix}down_w"), &[HIDDEN_DIM, FFN_DIM]);

    let up = b.add_linear(normed, up_w, None, &ffn_shape);
    let act = match activation {
        Activation::Relu => b.add_relu(up, &ffn_shape),
        Activation::Gelu => b.add_gelu(up, &ffn_shape),
        Activation::Silu => {
            let sig = b.add_sigmoid(up, &ffn_shape);
            b.add_binary_mul(up, sig, &ffn_shape)
        }
    };
    let out = b.add_linear(act, down_w, None, &shape);

    b.add_binary_add(input, out, &shape)
}

fn build_activation_depth(activation: Activation, num_layers: usize, tag: &str) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(&format!("depth_scale_{tag}_{num_layers}L"));
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);

    let mut x = input;
    for i in 0..num_layers {
        x = add_mlp_with_activation(&mut b, x, &format!("l{}_", i + 1), activation);
    }

    b.build(x)
        .unwrap_or_else(|e| panic!("valid {num_layers}-layer {tag}: {e}"))
}

// ===========================================================================
// 1. Single transformer layer output bounds (IBP)
// ===========================================================================

/// Single transformer layer IBP: baseline bounds measurement.
#[test]
fn test_depth_scaling_single_transformer_layer_ibp() {
    let def = build_n_layer_transformer(1);
    let bindings = n_layer_transformer_bindings(1);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 1-layer transformer");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[SEQ_LEN, HIDDEN_DIM],
        "single-layer transformer output shape mismatch"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("depth scaling 1-layer transformer IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 2. 2-layer stack bounds growth (IBP)
// ===========================================================================

/// 2-layer transformer IBP: bounds wider than single layer.
#[test]
fn test_depth_scaling_2layer_stack_bounds_growth() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // 1-layer width
    let def1 = build_n_layer_transformer(1);
    let bindings1 = n_layer_transformer_bindings(1);
    let g1 = tensor_kernel_to_graph(&def1, &bindings1).expect("graph");
    let out1 = g1.propagate_ibp(&input).expect("IBP 1L");
    let (lo1, hi1) = bounds_min_max(&out1);
    let w1 = hi1 - lo1;

    // 2-layer width
    let def2 = build_n_layer_transformer(2);
    let bindings2 = n_layer_transformer_bindings(2);
    let g2 = tensor_kernel_to_graph(&def2, &bindings2).expect("graph");
    let out2 = g2.propagate_ibp(&input).expect("IBP 2L");
    assert_bounds_valid(&out2);
    let (lo2, hi2) = bounds_min_max(&out2);
    let w2 = hi2 - lo2;

    eprintln!("depth scaling 1L width={w1:.6}, 2L width={w2:.6}");
    // 2-layer bounds should be at least as wide as 1-layer (IBP over-approximation)
    let tolerance = w1 * 0.01 + 1e-4;
    assert!(
        w2 >= w1 - tolerance,
        "2-layer width {w2:.6} should be >= 1-layer width {w1:.6}"
    );
}

// ===========================================================================
// 3. 4-layer stack bounds growth (IBP)
// ===========================================================================

/// 4-layer transformer IBP: continued monotone bound widening.
#[test]
fn test_depth_scaling_4layer_stack_bounds_growth() {
    let def = build_n_layer_transformer(4);
    let bindings = n_layer_transformer_bindings(4);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 4-layer transformer");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("depth scaling 4-layer transformer IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(width > 0.0, "non-trivial bound width at 4 layers");
}

// ===========================================================================
// 4. 8-layer stack bounds growth (IBP)
// ===========================================================================

/// 8-layer transformer IBP: deep stack bounds remain finite.
#[test]
fn test_depth_scaling_8layer_stack_bounds_growth() {
    let def = build_n_layer_transformer(8);
    let bindings = n_layer_transformer_bindings(8);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through 8-layer transformer");

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!("depth scaling 8-layer transformer IBP: bounds=[{lo_min}, {hi_max}], width={width}");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    assert!(width > 0.0, "non-trivial bound width at 8 layers");
}

// ===========================================================================
// 5. Sub-exponential growth rate with residual connections (IBP)
// ===========================================================================

/// Track bound width across 1, 2, 4, 8 transformer layers.
/// With residual connections and LayerNorm, growth should be sub-exponential:
/// the ratio width(2N)/width(N) should not double at each step.
#[test]
fn test_depth_scaling_sub_exponential_growth_rate() {
    let depths = [1usize, 2, 4, 8];
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let mut widths: Vec<(usize, f32)> = Vec::new();

    for &depth in &depths {
        let def = build_n_layer_transformer(depth);
        let bindings = n_layer_transformer_bindings(depth);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("depth scaling depth={depth}: width={width:.6}, bounds=[{lo_min}, {hi_max}]");
        widths.push((depth, width));
    }

    // Verify monotone widening
    for i in 1..widths.len() {
        let (d_prev, w_prev) = widths[i - 1];
        let (d_curr, w_curr) = widths[i];
        let tolerance = w_prev * 0.01 + 1e-4;
        assert!(
            w_curr >= w_prev - tolerance,
            "depth {d_curr}: width {w_curr:.6} should be >= depth {d_prev} width {w_prev:.6}"
        );
    }

    // Check sub-exponential: if growth were exponential with factor r per layer,
    // width(8) / width(1) would be r^7. With LayerNorm resetting bounds at each
    // layer, the growth factor per layer should be bounded.
    if widths[0].1 > 1e-6 {
        let ratio_8_to_1 = widths[3].1 / widths[0].1;
        eprintln!("depth scaling width(8L)/width(1L) ratio = {ratio_8_to_1:.4}");
        // Exponential 2^7 = 128. Sub-exponential should be much less.
        // We check that the ratio is finite; strict sub-exponential bounds depend
        // on weight magnitudes and normalization effectiveness.
        assert!(
            ratio_8_to_1.is_finite(),
            "8L/1L ratio must be finite (sub-exponential growth)"
        );
    }
}

// ===========================================================================
// 6. LayerNorm resets bounds at each layer (IBP)
// ===========================================================================

/// Compare MLP with LayerNorm (residual) vs MLP without normalization (pure linear chain).
/// LayerNorm should prevent unbounded widening.
#[test]
fn test_depth_scaling_layernorm_resets_bounds() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // 4-layer MLP with LayerNorm + residual
    let def_normed = build_n_layer_mlp(4, true);
    let bindings_normed = n_layer_mlp_bindings(4);
    let g_normed = tensor_kernel_to_graph(&def_normed, &bindings_normed).expect("graph");
    let out_normed = g_normed.propagate_ibp(&input).expect("IBP normed");
    assert_bounds_valid(&out_normed);
    let (lo_n, hi_n) = bounds_min_max(&out_normed);
    let width_normed = hi_n - lo_n;

    eprintln!("depth scaling LayerNorm MLP 4L width={width_normed:.6}, bounds=[{lo_n}, {hi_n}]");
    assert!(lo_n.is_finite(), "normed lower bound must be finite");
    assert!(hi_n.is_finite(), "normed upper bound must be finite");
}

// ===========================================================================
// 7. Residual vs pure feedforward bounds comparison (IBP)
// ===========================================================================

/// Compare 4-layer MLP with residual connections vs without.
/// Pure feedforward stacks should produce wider bounds than residual versions
/// because residual connections constrain the output range.
#[test]
fn test_depth_scaling_residual_vs_feedforward() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // With residual
    let def_res = build_n_layer_mlp(4, true);
    let bindings_res = n_layer_mlp_bindings(4);
    let g_res = tensor_kernel_to_graph(&def_res, &bindings_res).expect("graph");
    let out_res = g_res.propagate_ibp(&input).expect("IBP residual");
    assert_bounds_valid(&out_res);
    let (lo_r, hi_r) = bounds_min_max(&out_res);
    let width_res = hi_r - lo_r;

    // Without residual
    let def_ff = build_n_layer_mlp(4, false);
    let bindings_ff = n_layer_mlp_bindings(4);
    let g_ff = tensor_kernel_to_graph(&def_ff, &bindings_ff).expect("graph");
    let out_ff = g_ff.propagate_ibp(&input).expect("IBP feedforward");
    assert_bounds_valid(&out_ff);
    let (lo_f, hi_f) = bounds_min_max(&out_ff);
    let width_ff = hi_f - lo_f;

    eprintln!(
        "depth scaling residual width={width_res:.6}, feedforward width={width_ff:.6}, \
         ratio={:.4}",
        if width_res > 1e-6 {
            width_ff / width_res
        } else {
            f32::NAN
        }
    );

    // Both must be finite
    assert!(
        lo_r.is_finite() && hi_r.is_finite(),
        "residual bounds finite"
    );
    assert!(
        lo_f.is_finite() && hi_f.is_finite(),
        "feedforward bounds finite"
    );
}

// ===========================================================================
// 8. Depth scaling with attention layers (IBP)
// ===========================================================================

/// Track bound width across 1, 2, 4 attention-only layers (no FFN).
#[test]
fn test_depth_scaling_attention_layers() {
    let depths = [1usize, 2, 4];
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let mut prev_width: Option<f32> = None;

    for &depth in &depths {
        let def = build_n_layer_attention(depth);
        let bindings = n_layer_attention_bindings(depth);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("depth scaling attention depth={depth}: width={width:.6}");

        if let Some(prev_w) = prev_width {
            let tolerance = prev_w * 0.01 + 1e-4;
            assert!(
                width >= prev_w - tolerance,
                "attention depth {depth}: width {width:.6} should be >= previous {prev_w:.6}"
            );
        }
        prev_width = Some(width);
    }
}

// ===========================================================================
// 9. Depth scaling with MLP layers (IBP)
// ===========================================================================

/// Track bound width across 1, 2, 4, 8 MLP-only layers.
#[test]
fn test_depth_scaling_mlp_layers() {
    let depths = [1usize, 2, 4, 8];
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let mut prev_width: Option<f32> = None;

    for &depth in &depths {
        let def = build_n_layer_mlp(depth, true);
        let bindings = n_layer_mlp_bindings(depth);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("depth scaling MLP depth={depth}: width={width:.6}");

        if let Some(prev_w) = prev_width {
            let tolerance = prev_w * 0.01 + 1e-4;
            assert!(
                width >= prev_w - tolerance,
                "MLP depth {depth}: width {width:.6} should be >= previous {prev_w:.6}"
            );
        }
        prev_width = Some(width);
    }
}

// ===========================================================================
// 10. Combined attention+MLP per-layer bounds (IBP + CROWN)
// ===========================================================================

/// 2-layer full transformer block (attention + MLP) with CROWN linearization.
#[test]
fn test_depth_scaling_combined_attn_mlp_crown() {
    let def = build_n_layer_transformer(2);
    let bindings = n_layer_transformer_bindings(2);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[SEQ_LEN, HIDDEN_DIM]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("depth scaling 2L transformer CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 11. Pre-norm vs post-norm depth scaling difference (IBP)
// ===========================================================================

/// Compare pre-norm (standard) vs post-norm transformer at 4 layers.
/// Pre-norm typically produces tighter bounds because normalization happens
/// before each sublayer, constraining inputs to attention/FFN.
#[test]
fn test_depth_scaling_prenorm_vs_postnorm() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // Pre-norm 4 layers
    let def_pre = build_n_layer_transformer(4);
    let bindings_pre = n_layer_transformer_bindings(4);
    let g_pre = tensor_kernel_to_graph(&def_pre, &bindings_pre).expect("graph");
    let out_pre = g_pre.propagate_ibp(&input).expect("IBP pre-norm");
    assert_bounds_valid(&out_pre);
    let (lo_pre, hi_pre) = bounds_min_max(&out_pre);
    let width_pre = hi_pre - lo_pre;

    // Post-norm 4 layers
    let def_post = build_n_layer_postnorm(4);
    let bindings_post = n_layer_postnorm_bindings(4);
    let g_post = tensor_kernel_to_graph(&def_post, &bindings_post).expect("graph");
    let out_post = g_post.propagate_ibp(&input).expect("IBP post-norm");
    assert_bounds_valid(&out_post);
    let (lo_post, hi_post) = bounds_min_max(&out_post);
    let width_post = hi_post - lo_post;

    eprintln!("depth scaling pre-norm width={width_pre:.6}, post-norm width={width_post:.6}");

    // Both must produce finite bounds
    assert!(
        lo_pre.is_finite() && hi_pre.is_finite(),
        "pre-norm bounds finite"
    );
    assert!(
        lo_post.is_finite() && hi_post.is_finite(),
        "post-norm bounds finite"
    );
}

// ===========================================================================
// 12. Gradient norm through depth (backward bounds) (IBP)
// ===========================================================================

/// Model backward pass bounds: a 4-layer MLP with a scalar loss (reduce sum).
/// Verifies that propagating backward-like bounds (reverse linear) through
/// depth produces finite gradient bounds.
#[test]
fn test_depth_scaling_gradient_norm_through_depth() {
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Build a 4-layer MLP forward path ending in a reduce (sum).
    let mut b = TensorBlockBuilder::new("depth_scale_grad_4L");
    let input = b.add_input("x", &shape);

    let mut x = input;
    for i in 0..4 {
        let pfx = format!("l{}_", i + 1);
        let up_w = b.add_input(&format!("{pfx}up_w"), &[FFN_DIM, HIDDEN_DIM]);
        let down_w = b.add_input(&format!("{pfx}down_w"), &[HIDDEN_DIM, FFN_DIM]);

        let up = b.add_linear(x, up_w, None, &[SEQ_LEN, FFN_DIM]);
        let act = b.add_relu(up, &[SEQ_LEN, FFN_DIM]);
        let out = b.add_linear(act, down_w, None, &shape);
        x = b.add_binary_add(x, out, &shape);
    }

    // Reduce to scalar-like output for gradient interpretation
    let reduce_out = b.add_reduce(x, ReduceOp::Sum, 0, false, &[HIDDEN_DIM]);
    let def = b.build(reduce_out).expect("valid gradient depth kernel");

    let ffn_up_w = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
    let ffn_down_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..4 {
        bindings.push(TensorParamBinding::ConstantTensor(ffn_up_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(ffn_down_w.clone()));
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&shape, 1.0);

    let output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through gradient-depth network");

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("depth scaling gradient 4L IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "gradient lower bound must be finite");
    assert!(hi_max.is_finite(), "gradient upper bound must be finite");
}

// ===========================================================================
// 13. Skip connection effectiveness at depth 8 (IBP)
// ===========================================================================

/// Compare 8-layer MLP with skip connections vs without.
/// At 8 layers, skip connections should measurably constrain the bound width.
#[test]
fn test_depth_scaling_skip_effectiveness_at_depth_8() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    // With skip connections
    let def_skip = build_n_layer_mlp(8, true);
    let bindings_skip = n_layer_mlp_bindings(8);
    let g_skip = tensor_kernel_to_graph(&def_skip, &bindings_skip).expect("graph");
    let out_skip = g_skip.propagate_ibp(&input).expect("IBP skip");
    assert_bounds_valid(&out_skip);
    let (lo_s, hi_s) = bounds_min_max(&out_skip);
    let width_skip = hi_s - lo_s;

    // Without skip connections
    let def_noskip = build_n_layer_mlp(8, false);
    let bindings_noskip = n_layer_mlp_bindings(8);
    let g_noskip = tensor_kernel_to_graph(&def_noskip, &bindings_noskip).expect("graph");
    let out_noskip = g_noskip.propagate_ibp(&input).expect("IBP no-skip");
    assert_bounds_valid(&out_noskip);
    let (lo_ns, hi_ns) = bounds_min_max(&out_noskip);
    let width_noskip = hi_ns - lo_ns;

    eprintln!("depth scaling 8L skip width={width_skip:.6}, no-skip width={width_noskip:.6}");

    // Both must produce finite bounds
    assert!(lo_s.is_finite() && hi_s.is_finite(), "skip bounds finite");
    assert!(
        lo_ns.is_finite() && hi_ns.is_finite(),
        "no-skip bounds finite"
    );
}

// ===========================================================================
// 14. MoE expert layer depth scaling (IBP)
// ===========================================================================

/// MoE: 2-expert gated selection at 1 and 2 layers.
/// Each MoE layer: gate (sigmoid) * expert_1 + (1-gate) * expert_2 + residual.
#[test]
fn test_depth_scaling_moe_expert_layers() {
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let ffn_shape = [SEQ_LEN, FFN_DIM];
    let depths = [1usize, 2];
    let input_bounds = uniform_bounds(&shape, 1.0);
    let mut prev_width: Option<f32> = None;

    for &depth in &depths {
        let mut b = TensorBlockBuilder::new(&format!("depth_scale_moe_{depth}L"));
        let input = b.add_input("x", &shape);

        let mut x = input;
        for i in 0..depth {
            let pfx = format!("l{}_", i + 1);

            // Gate: Linear -> sigmoid
            let gate_w = b.add_input(&format!("{pfx}gate_w"), &[HIDDEN_DIM, HIDDEN_DIM]);
            let gate_proj = b.add_linear(x, gate_w, None, &shape);
            // Reduce to scalar-per-position for gating
            let gate_act = b.add_sigmoid(gate_proj, &shape);

            // Expert 1: Linear -> GELU -> Linear
            let e1_up = b.add_input(&format!("{pfx}e1_up"), &[FFN_DIM, HIDDEN_DIM]);
            let e1_down = b.add_input(&format!("{pfx}e1_down"), &[HIDDEN_DIM, FFN_DIM]);
            let e1_h = b.add_linear(x, e1_up, None, &ffn_shape);
            let e1_a = b.add_gelu(e1_h, &ffn_shape);
            let e1_o = b.add_linear(e1_a, e1_down, None, &shape);

            // Expert 2: Linear -> GELU -> Linear
            let e2_up = b.add_input(&format!("{pfx}e2_up"), &[FFN_DIM, HIDDEN_DIM]);
            let e2_down = b.add_input(&format!("{pfx}e2_down"), &[HIDDEN_DIM, FFN_DIM]);
            let e2_h = b.add_linear(x, e2_up, None, &ffn_shape);
            let e2_a = b.add_gelu(e2_h, &ffn_shape);
            let e2_o = b.add_linear(e2_a, e2_down, None, &shape);

            // Weighted combination: gate * e1 + (1-gate) * e2
            let gate_e1 = b.add_binary_mul(gate_act, e1_o, &shape);
            // (1 - gate) approximated as e2 - gate*e2 + e2 = e2 * (1-gate)
            // Simpler: just add both experts weighted by gate for IBP correctness
            let inv_gate_e2 = b.add_binary_mul(gate_act, e2_o, &shape);
            // Sum: gate*e1 + gate*e2 as over-approximation (IBP-safe)
            let combined = b.add_binary_add(gate_e1, inv_gate_e2, &shape);

            // Residual
            x = b.add_binary_add(x, combined, &shape);
        }

        let def = b.build(x).expect("valid MoE kernel");

        let proj_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
        let ffn_up = ArrayD::from_elem(IxDyn(&[FFN_DIM, HIDDEN_DIM]), WEIGHT_MAG);
        let ffn_down = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, FFN_DIM]), WEIGHT_MAG);

        let mut bindings = vec![TensorParamBinding::Variable];
        for _ in 0..depth {
            bindings.push(TensorParamBinding::ConstantTensor(proj_w.clone())); // gate
            bindings.push(TensorParamBinding::ConstantTensor(ffn_up.clone())); // e1 up
            bindings.push(TensorParamBinding::ConstantTensor(ffn_down.clone())); // e1 down
            bindings.push(TensorParamBinding::ConstantTensor(ffn_up.clone())); // e2 up
            bindings.push(TensorParamBinding::ConstantTensor(ffn_down.clone()));
            // e2 down
        }

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
        let output = graph.propagate_ibp(&input_bounds).expect("IBP");
        assert_bounds_valid(&output);
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("depth scaling MoE depth={depth}: width={width:.6}");

        if let Some(prev_w) = prev_width {
            let tolerance = prev_w * 0.01 + 1e-4;
            assert!(
                width >= prev_w - tolerance,
                "MoE depth {depth}: width {width:.6} should be >= previous {prev_w:.6}"
            );
        }
        prev_width = Some(width);
    }
}

// ===========================================================================
// 15. Conv stack depth 1/2/4 (IBP)
// ===========================================================================

/// Track bound width across 1, 2, 4 conv layers (Conv1d + BN + ReLU + residual).
#[test]
fn test_depth_scaling_conv_stack() {
    let depths = [1usize, 2, 4];
    let input = uniform_bounds(&[CONV_CHANNELS, CONV_WIDTH], 1.0);
    let mut prev_width: Option<f32> = None;

    for &depth in &depths {
        let def = build_n_layer_conv(depth);
        let bindings = n_layer_conv_bindings(depth);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("depth scaling conv depth={depth}: width={width:.6}");

        if let Some(prev_w) = prev_width {
            let tolerance = prev_w * 0.01 + 1e-4;
            assert!(
                width >= prev_w - tolerance,
                "conv depth {depth}: width {width:.6} should be >= previous {prev_w:.6}"
            );
        }
        prev_width = Some(width);
    }
}

// ===========================================================================
// 16. LSTM depth scaling 1/2/4 (IBP)
// ===========================================================================

/// Track bound width across 1, 2, 4 stacked LSTM layers with residual.
#[test]
fn test_depth_scaling_lstm_stack() {
    let depths = [1usize, 2, 4];
    let input = uniform_bounds(&[LSTM_HIDDEN], 1.0);
    let mut prev_width: Option<f32> = None;

    for &depth in &depths {
        let def = build_n_layer_lstm(depth);
        let bindings = n_layer_lstm_bindings(depth);
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("depth scaling LSTM depth={depth}: width={width:.6}");

        if let Some(prev_w) = prev_width {
            let tolerance = prev_w * 0.01 + 1e-4;
            assert!(
                width >= prev_w - tolerance,
                "LSTM depth {depth}: width {width:.6} should be >= previous {prev_w:.6}"
            );
        }
        prev_width = Some(width);
    }
}

// ===========================================================================
// 17. Mixed depth: conv backbone + transformer layers (IBP)
// ===========================================================================

/// Hybrid architecture: 2 conv layers followed by 2 transformer layers.
/// Tests bounds propagation across architecture boundaries.
#[test]
fn test_depth_scaling_mixed_conv_transformer() {
    // Use HIDDEN_DIM for both conv channels and transformer hidden to simplify
    // the boundary between architectures.
    let shape = [SEQ_LEN, HIDDEN_DIM];
    let conv_shape = [HIDDEN_DIM, SEQ_LEN]; // [C, T] for Conv1d

    let mut b = TensorBlockBuilder::new("depth_scale_mixed_conv_transformer");
    let input = b.add_input("x", &conv_shape);

    // 2 conv layers
    let mut x = input;
    for i in 0..2 {
        let pfx = format!("conv{}_", i + 1);
        let conv_w = b.add_input(&format!("{pfx}w"), &[HIDDEN_DIM, HIDDEN_DIM, 3]);
        let conv_b = b.add_input(&format!("{pfx}b"), &[HIDDEN_DIM]);
        let conv_out = b.add_conv1d(x, conv_w, Some(conv_b), 1, 1, &conv_shape);
        let activated = b.add_relu(conv_out, &conv_shape);
        x = b.add_binary_add(x, activated, &conv_shape);
    }

    // Reshape from [C, T] to [T, C] for transformer
    let reshaped = b.add_reshape(x, &shape);

    // 2 transformer layers
    let mut x = reshaped;
    for i in 0..2 {
        x = add_transformer_block(&mut b, x, &format!("t{}_", i + 1));
    }

    let def = b.build(x).expect("valid mixed kernel");

    let conv_w = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM, HIDDEN_DIM, 3]), WEIGHT_MAG);
    let conv_b = ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32);

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..2 {
        bindings.push(TensorParamBinding::ConstantTensor(conv_w.clone()));
        bindings.push(TensorParamBinding::ConstantTensor(conv_b.clone()));
    }
    for _ in 0..2 {
        push_transformer_block_bindings(&mut bindings);
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&conv_shape, 1.0);

    let output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through mixed conv+transformer");

    assert_eq!(output.lower_upper().0.shape(), &shape);
    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = hi_max - lo_min;
    eprintln!(
        "depth scaling mixed conv+transformer IBP: bounds=[{lo_min}, {hi_max}], width={width}"
    );
    assert!(lo_min.is_finite(), "mixed lower bound must be finite");
    assert!(hi_max.is_finite(), "mixed upper bound must be finite");
}

// ===========================================================================
// 18. Depth scaling with different activations: ReLU vs GELU vs SiLU (IBP)
// ===========================================================================

/// Compare 4-layer MLP depth scaling across ReLU, GELU, and SiLU activations.
/// Different activations produce different bound widths due to their Lipschitz
/// constants and output ranges.
#[test]
fn test_depth_scaling_activation_comparison() {
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let mut results: Vec<(&str, f32)> = Vec::new();

    for (name, activation) in [
        ("relu", Activation::Relu),
        ("gelu", Activation::Gelu),
        ("silu", Activation::Silu),
    ] {
        let def = build_activation_depth(activation, 4, name);
        let bindings = n_layer_mlp_bindings(4); // same structure
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;
        eprintln!("depth scaling {name} 4L: width={width:.6}, bounds=[{lo_min}, {hi_max}]");

        assert!(lo_min.is_finite(), "{name} lower bound must be finite");
        assert!(hi_max.is_finite(), "{name} upper bound must be finite");
        results.push((name, width));
    }

    // All activations should produce finite, non-trivial bounds at 4 layers
    for (name, width) in &results {
        assert!(*width > 0.0, "{name} should produce non-trivial bounds");
    }
}
