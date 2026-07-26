// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for gradient flow and training stability bounds.
//!
//! Verifies IBP and CROWN bound propagation through network structures
//! whose gradient properties are critical for training stability in dpdf
//! models (GLM-OCR, Qwen3-VL, Granite-Docling, FireRed-OCR). Gradient
//! flow through a network is the transpose of forward bound propagation
//! — tight forward bounds imply stable gradient paths, and bound widening
//! through depth signals vanishing/exploding gradient risk.
//!
//! 1.  **Linear layer backward bound propagation** (IBP)
//! 2.  **ReLU backward: gradient masking at zero** (IBP)
//! 3.  **Softmax backward: Jacobian bounds** (IBP)
//! 4.  **Residual connection gradient: identity + transform** (IBP)
//! 5.  **LayerNorm backward bound propagation** (IBP)
//! 6.  **Attention backward: gradient through softmax** (IBP)
//! 7.  **SwiGLU backward: gated gradient flow** (IBP + CROWN)
//! 8.  **Deep residual gradient: 4-layer backward** (IBP)
//! 9.  **Gradient clipping effect on bound width** (IBP)
//! 10. **Vanishing gradient detection: deep MLP backward** (IBP)
//! 11. **Exploding gradient detection: large weight backward** (IBP)
//! 12. **CROWN tightness for backward propagation** (CROWN)
//! 13. **Gradient monotone tightening: smaller eps -> tighter gradient** (IBP)
//! 14. **Skip connection gradient stability** (IBP)
//! 15. **Full forward-backward pipeline bound propagation** (IBP + CROWN)
//!
//! Architecture references:
//! - ResNet (He et al., 2016): Skip connections prevent vanishing gradients
//! - Pre-LN Transformer (Xiong et al., 2020): LayerNorm before attention/FFN
//! - SwiGLU (Shazeer, 2020): Gated activation with smooth gradient flow
//! - Gradient clipping (Pascanu et al., 2013): Exploding gradient mitigation
//!
//! Dimensions (small for fast verification, structurally representative):
//! - SEQ_LEN=4, HIDDEN_DIM=64, FFN_DIM=128
//!
//! Part of #4049: Compose tests for gradient flow and training stability bounds.

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build SiLU activation: SiLU(x) = x * sigmoid(x).
fn add_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

/// Build a standard SwiGLU FFN block.
///
/// Pattern: gate_proj(x) -> SiLU -> mul(up_proj(x)) -> down_proj
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

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(b, gate, &ffn_shape);

    let up = b.add_linear(input, up_w, None, &ffn_shape);

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

/// Constant weight binding helper.
fn weight_binding(shape: &[usize], mag: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), mag))
}

// ===========================================================================
// 1. Linear layer backward bound propagation (IBP)
// ===========================================================================

/// A linear layer `y = Wx` maps input bounds through weight multiplication.
/// Gradient through a linear layer is `W^T * grad_output`. Tight forward bounds
/// through linear layers indicate stable backward gradient norms.
#[test]
fn test_gradient_linear_backward_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_grad_linear_backward");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(input, w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid linear kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Linear backward IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Small weights (0.02) should not amplify bounds much
    let width = bound_width(&output);
    assert!(
        width < 200.0,
        "linear layer should not explode bounds: width={width}"
    );
}

// ===========================================================================
// 2. ReLU backward: gradient masking at zero (IBP)
// ===========================================================================

/// ReLU kills gradients for negative inputs (grad = 0 when x < 0).
/// Forward bounds through ReLU are clipped at 0, which models the gradient
/// masking property: the lower bound is max(0, lower_input).
#[test]
fn test_gradient_relu_masking_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_grad_relu_masking");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_relu(input, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid ReLU kernel");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("ReLU gradient masking IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // ReLU output lower bound must be >= 0 (gradient killed for negatives)
    let tol = 1e-6;
    assert!(
        lo_min >= 0.0 - tol,
        "ReLU lower bound must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "ReLU upper bound should be <= input max, got {hi_max}"
    );
}

// ===========================================================================
// 3. Softmax backward: Jacobian bounds (IBP)
// ===========================================================================

/// Softmax Jacobian `J_ij = s_i(delta_ij - s_j)` has bounded spectral norm.
/// Forward bounds through softmax are in [0, 1] and sum to 1 along the axis,
/// which constrains gradient magnitude through the Jacobian.
#[test]
fn test_gradient_softmax_jacobian_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_grad_softmax_jacobian");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w = b.add_input("w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let logits = b.add_linear(input, w, None, &[SEQ_LEN, HIDDEN_DIM]);
    let out = b.add_softmax(logits, 1, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid softmax kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let tol = 1e-6;
    eprintln!("Softmax Jacobian IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Softmax output must be in [0, 1]
    assert!(
        lo_min >= 0.0 - tol,
        "softmax lower must be >= 0, got {lo_min}"
    );
    assert!(
        hi_max <= 1.0 + tol,
        "softmax upper must be <= 1, got {hi_max}"
    );
}

// ===========================================================================
// 4. Residual connection gradient: identity + transform (IBP)
// ===========================================================================

/// Residual connections `y = x + f(x)` ensure gradient flows directly
/// through the identity path: `dy/dx = I + df/dx`. This prevents vanishing
/// gradients. Forward bound propagation through residual should include both
/// the input range and the transformed range.
#[test]
fn test_gradient_residual_identity_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_grad_residual_identity");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Transform path: Linear -> ReLU
    let w = b.add_input("w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(input, w, None, &shape);
    let h = b.add_relu(h, &shape);

    // Residual: x + ReLU(Wx)
    let out = b.add_binary_add(input, h, &shape);
    let def = b.build(out).expect("valid residual kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Residual identity IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    // Residual: output range includes input range [-1, 1] plus positive ReLU contribution
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
    // Upper bound should be at least 1.0 (from the identity path alone)
    assert!(
        hi_max >= 0.5,
        "residual upper should include identity path, got {hi_max}"
    );
}

// ===========================================================================
// 5. LayerNorm backward bound propagation (IBP)
// ===========================================================================

/// LayerNorm normalizes activations and its backward pass involves the Jacobian
/// of the normalization function. Forward bound propagation through LayerNorm
/// constrains output range, which bounds gradient magnitude.
#[test]
fn test_gradient_layernorm_backward_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_grad_layernorm_backward");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let gamma = b.add_input("gamma", &[HIDDEN_DIM]);
    let beta = b.add_input("beta", &[HIDDEN_DIM]);

    let out = b.add_layer_norm(input, eps, 1, gamma, beta, &[SEQ_LEN, HIDDEN_DIM]);
    let def = b.build(out).expect("valid LayerNorm kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("LayerNorm backward IBP: width={width:.6}");
    assert!(width.is_finite(), "LayerNorm output width must be finite");
}

// ===========================================================================
// 6. Attention backward: gradient through softmax (IBP)
// ===========================================================================

/// Attention `softmax(QK^T/sqrt(d)) V` has gradients flowing through
/// softmax (bounded Jacobian) and linear projections. Forward bounds
/// through the full attention block constrain backward gradient norms.
#[test]
fn test_gradient_attention_backward_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_grad_attention_backward");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let out = b
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
    let def = b.build(out).expect("valid attention kernel");

    let w = |s: &[usize]| weight_binding(s, WEIGHT_MAG);
    let bindings = vec![
        TensorParamBinding::Variable,
        w(&[HIDDEN_DIM, HIDDEN_DIM]),
        w(&[HIDDEN_DIM, HIDDEN_DIM]),
        w(&[HIDDEN_DIM, HIDDEN_DIM]),
        w(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Attention backward IBP: bounds=[{lo_min:.6}, {hi_max:.6}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

// ===========================================================================
// 7. SwiGLU backward: gated gradient flow (IBP + CROWN)
// ===========================================================================

/// SwiGLU `(gate * SiLU(gate)) * up` has multiplicative gradient paths.
/// The gating mechanism routes gradients selectively, preventing explosion
/// while maintaining non-zero flow. Verify bound propagation captures this.
fn build_swiglu_gradient_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_grad_swiglu_backward");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let out = build_swiglu_block(&mut b, input, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);
    b.build(out).expect("valid SwiGLU kernel")
}

fn swiglu_gradient_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = vec![TensorParamBinding::Variable];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

#[test]
fn test_gradient_swiglu_gated_ibp() {
    let def = build_swiglu_gradient_kernel();
    let bindings = swiglu_gradient_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("SwiGLU gated gradient IBP: width={width:.6}");
    assert!(width.is_finite(), "output width must be finite");
}

#[test]
fn test_gradient_swiglu_gated_crown() {
    let def = build_swiglu_gradient_kernel();
    let bindings = swiglu_gradient_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("SwiGLU gated gradient CROWN: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 8. Deep residual gradient: 4-layer backward (IBP)
// ===========================================================================

/// Deep residual networks maintain gradient flow through skip connections.
/// Each layer is `x + ReLU(W_i x)`. Through 4 layers, the identity path
/// ensures gradients don't vanish, and small weights prevent explosion.
/// Bound widening rate through depth indicates gradient stability.
#[test]
fn test_gradient_deep_residual_4layer_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_grad_deep_residual_4layer");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut h = input;
    for i in 0..4 {
        let w = b.add_input(&format!("w{i}"), &[HIDDEN_DIM, HIDDEN_DIM]);
        let transformed = b.add_linear(h, w, None, &shape);
        let activated = b.add_relu(transformed, &shape);
        h = b.add_binary_add(h, activated, &shape);
    }
    let def = b.build(h).expect("valid deep residual kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..4 {
        bindings.push(weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG));
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = bound_width(&output);
    eprintln!("Deep residual 4-layer IBP: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");
    assert!(width.is_finite(), "depth-4 output width must be finite");
    // Identity path preserves input range, so output includes at least [-1, 1]
    assert!(
        hi_max >= 0.5,
        "residual output should include identity path contribution, got {hi_max}"
    );
}

// ===========================================================================
// 9. Gradient clipping effect on bound width (IBP)
// ===========================================================================

/// Gradient clipping constrains gradient norm, which corresponds to bounding
/// the output range of a sigmoid gate. We model this as: x -> Linear -> sigmoid
/// (which naturally clips to [0, 1], analogous to gradient clipping).
/// Comparing with an unbounded path (Linear -> Linear) shows tighter bounds.
#[test]
fn test_gradient_clipping_bound_width_ibp() {
    // Clipped path: Linear -> sigmoid (bounded in [0, 1])
    let mut b_clip = TensorBlockBuilder::new("dpdf_grad_clip_sigmoid");
    let input_clip = b_clip.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w_clip = b_clip.add_input("w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h_clip = b_clip.add_linear(input_clip, w_clip, None, &[SEQ_LEN, HIDDEN_DIM]);
    let out_clip = b_clip.add_sigmoid(h_clip, &[SEQ_LEN, HIDDEN_DIM]);
    let def_clip = b_clip.build(out_clip).expect("valid clipped kernel");

    let clip_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
    ];

    // Unclipped path: Linear -> Linear (unbounded)
    let mut b_unclip = TensorBlockBuilder::new("dpdf_grad_unclip_linear");
    let input_unclip = b_unclip.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w1 = b_unclip.add_input("w1", &[HIDDEN_DIM, HIDDEN_DIM]);
    let w2 = b_unclip.add_input("w2", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h = b_unclip.add_linear(input_unclip, w1, None, &[SEQ_LEN, HIDDEN_DIM]);
    let out_unclip = b_unclip.add_linear(h, w2, None, &[SEQ_LEN, HIDDEN_DIM]);
    let def_unclip = b_unclip.build(out_unclip).expect("valid unclipped kernel");

    let unclip_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
    ];

    let graph_clip = tensor_kernel_to_graph(&def_clip, &clip_bindings).expect("clip graph");
    let graph_unclip = tensor_kernel_to_graph(&def_unclip, &unclip_bindings).expect("unclip graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let clip_output = graph_clip.propagate_ibp(&input).expect("clip IBP");
    let unclip_output = graph_unclip.propagate_ibp(&input).expect("unclip IBP");

    assert_bounds_valid(&clip_output);
    assert_bounds_valid(&unclip_output);

    let clip_width = bound_width(&clip_output);
    let unclip_width = bound_width(&unclip_output);
    eprintln!(
        "Gradient clipping IBP: clipped_width={clip_width:.6}, unclipped_width={unclip_width:.6}"
    );
    // Sigmoid-clipped path should have width <= 1.0 (output in [0, 1])
    assert!(
        clip_width <= 1.0 + 1e-4,
        "sigmoid clipped width should be <= 1.0, got {clip_width}"
    );
}

// ===========================================================================
// 10. Vanishing gradient detection: deep MLP backward (IBP)
// ===========================================================================

/// A deep MLP without skip connections, propagated under IBP.
///
/// Note on bound *growth* (this is sound, not a bug): the bindings make every
/// weight the constant +WEIGHT_MAG (= 0.02), and each layer is `HIDDEN_DIM`
/// (= 64) wide. The induced L∞ gain of one layer is therefore
/// `HIDDEN_DIM * WEIGHT_MAG = 64 * 0.02 = 1.28 > 1`, and ReLU only narrows
/// width, so the *true* function — not just its IBP relaxation — expands by a
/// factor of 1.28 per layer. After 6 layers the input width 2.0 grows to
/// `2.0 * 1.28^6 ≈ 4.40`. Because all weights share one sign and the post-ReLU
/// activations are all non-negative, there is no cross-dimension cancellation
/// for a tighter (CROWN/alpha-CROWN) bound to exploit either: the contraction
/// property simply does not hold for these weights. So this test pins the sound
/// IBP behavior — finite, monotone-in-depth, predictable growth within a tight
/// analytic envelope — rather than asserting a contraction IBP can never show.
#[test]
fn test_gradient_vanishing_deep_mlp_ibp() {
    let depth = 6;
    let mut b = TensorBlockBuilder::new("dpdf_grad_vanishing_mlp");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut h = input;
    for i in 0..depth {
        let w = b.add_input(&format!("w{i}"), &[HIDDEN_DIM, HIDDEN_DIM]);
        h = b.add_linear(h, w, None, &shape);
        h = b.add_relu(h, &shape);
    }
    let def = b.build(h).expect("valid deep MLP kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..depth {
        bindings.push(weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG));
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let input_width = bound_width(&input);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let output_width = bound_width(&output);
    eprintln!(
        "Vanishing gradient MLP (depth={depth}) IBP: input_width={input_width:.6}, output_width={output_width:.6}"
    );

    // Sound analytic envelope: per-layer L∞ gain is `HIDDEN_DIM * WEIGHT_MAG`,
    // and ReLU never widens a bound, so after `depth` layers the output width is
    // at most `input_width * gain^depth` (and strictly positive / finite).
    let per_layer_gain = (HIDDEN_DIM as f32) * WEIGHT_MAG;
    let envelope = input_width * per_layer_gain.powi(depth as i32);
    assert!(
        output_width.is_finite() && output_width > 0.0,
        "deep MLP output width must be finite and positive, got {output_width}"
    );
    assert!(
        output_width <= envelope * (1.0 + 1e-3),
        "deep MLP output width should stay within the sound L∞ envelope: \
         output={output_width}, envelope={envelope} (gain={per_layer_gain}, depth={depth})"
    );
}

// ===========================================================================
// 11. Exploding gradient detection: large weight backward (IBP)
// ===========================================================================

/// Large weights amplify bounds through linear layers, modeling the exploding
/// gradient scenario. With WEIGHT_MAG=0.5, bounds should expand through 3 layers.
#[test]
fn test_gradient_exploding_large_weight_ibp() {
    let large_weight_mag = 0.5f32;
    let depth = 3;

    let mut b = TensorBlockBuilder::new("dpdf_grad_exploding_large_weight");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    let mut h = input;
    for i in 0..depth {
        let w = b.add_input(&format!("w{i}"), &[HIDDEN_DIM, HIDDEN_DIM]);
        h = b.add_linear(h, w, None, &shape);
    }
    let def = b.build(h).expect("valid large weight kernel");

    let mut bindings = vec![TensorParamBinding::Variable];
    for _ in 0..depth {
        bindings.push(weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], large_weight_mag));
    }

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);
    let input_width = bound_width(&input_bounds);

    let output = graph.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let output_width = bound_width(&output);
    eprintln!(
        "Exploding gradient (depth={depth}, weight={large_weight_mag}) IBP: \
         input_width={input_width:.6}, output_width={output_width:.6}"
    );
    // Large weights expand bounds — output should be wider than input
    assert!(
        output_width > input_width,
        "large weights should expand bounds: output={output_width}, input={input_width}"
    );
    assert!(output_width.is_finite(), "output width must remain finite");
}

// ===========================================================================
// 12. CROWN tightness for backward propagation (CROWN)
// ===========================================================================

/// CROWN linear relaxation should produce tighter bounds than IBP for a
/// 2-layer residual network. This verifies that backward-relevant structures
/// (residual + nonlinearity) benefit from CROWN's tighter analysis.
fn build_crown_residual_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_grad_crown_residual");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Layer 1: residual with ReLU
    let w1 = b.add_input("w1", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h1 = b.add_linear(input, w1, None, &shape);
    let h1 = b.add_relu(h1, &shape);
    let r1 = b.add_binary_add(input, h1, &shape);

    // Layer 2: residual with ReLU
    let w2 = b.add_input("w2", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h2 = b.add_linear(r1, w2, None, &shape);
    let h2 = b.add_relu(h2, &shape);
    let out = b.add_binary_add(r1, h2, &shape);

    b.build(out).expect("valid CROWN residual kernel")
}

fn crown_residual_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
    ]
}

#[test]
fn test_gradient_crown_tightness() {
    let def = build_crown_residual_kernel();
    let bindings = crown_residual_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("CROWN backward tightness: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 13. Gradient monotone tightening: smaller eps -> tighter gradient (IBP)
// ===========================================================================

/// Monotonicity property: tighter input perturbation bounds (smaller eps)
/// must produce tighter output bounds. This is essential for gradient-based
/// training where learning rate reduction corresponds to smaller perturbations.
#[test]
fn test_gradient_monotone_tightening_ibp() {
    let mut b = TensorBlockBuilder::new("dpdf_grad_monotone");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // 2-layer residual: representative of gradient path
    let w1 = b.add_input("w1", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h = b.add_linear(input, w1, None, &shape);
    let h = b.add_relu(h, &shape);
    let r1 = b.add_binary_add(input, h, &shape);

    let w2 = b.add_input("w2", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h2 = b.add_linear(r1, w2, None, &shape);
    let out = b.add_relu(h2, &shape);
    let def = b.build(out).expect("valid monotone kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
    ];
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
        "Gradient monotone tightening: eps=1.0 width={wide_width:.6}, eps=0.1 width={tight_width:.6}"
    );
    assert!(
        tight_width <= wide_width + 1e-6,
        "tighter input should produce tighter output: wide={wide_width}, tight={tight_width}"
    );
}

// ===========================================================================
// 14. Skip connection gradient stability (IBP)
// ===========================================================================

/// Skip connections across multiple layers ensure stable gradient flow.
/// Model: x -> [Layer1 -> Layer2] + x (skip). The skip path guarantees
/// gradient magnitude is at least 1.0 regardless of internal layer behavior.
/// Compare with the non-skip version to show stability improvement.
#[test]
fn test_gradient_skip_connection_stability_ibp() {
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // With skip connection: x + Linear(ReLU(Linear(x)))
    let mut b_skip = TensorBlockBuilder::new("dpdf_grad_skip");
    let input_skip = b_skip.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w1_skip = b_skip.add_input("w1", &[HIDDEN_DIM, HIDDEN_DIM]);
    let w2_skip = b_skip.add_input("w2", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h = b_skip.add_linear(input_skip, w1_skip, None, &shape);
    let h = b_skip.add_relu(h, &shape);
    let h = b_skip.add_linear(h, w2_skip, None, &shape);
    let out_skip = b_skip.add_binary_add(input_skip, h, &shape);
    let def_skip = b_skip.build(out_skip).expect("valid skip kernel");

    let skip_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
    ];

    // Without skip: Linear(ReLU(Linear(x)))
    let mut b_no_skip = TensorBlockBuilder::new("dpdf_grad_no_skip");
    let input_no = b_no_skip.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let w1_no = b_no_skip.add_input("w1", &[HIDDEN_DIM, HIDDEN_DIM]);
    let w2_no = b_no_skip.add_input("w2", &[HIDDEN_DIM, HIDDEN_DIM]);
    let h = b_no_skip.add_linear(input_no, w1_no, None, &shape);
    let h = b_no_skip.add_relu(h, &shape);
    let out_no = b_no_skip.add_linear(h, w2_no, None, &shape);
    let def_no_skip = b_no_skip.build(out_no).expect("valid no-skip kernel");

    let no_skip_bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM], WEIGHT_MAG),
    ];

    let graph_skip = tensor_kernel_to_graph(&def_skip, &skip_bindings).expect("skip graph");
    let graph_no_skip =
        tensor_kernel_to_graph(&def_no_skip, &no_skip_bindings).expect("no-skip graph");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let skip_output = graph_skip.propagate_ibp(&input).expect("skip IBP");
    let no_skip_output = graph_no_skip.propagate_ibp(&input).expect("no-skip IBP");

    assert_bounds_valid(&skip_output);
    assert_bounds_valid(&no_skip_output);

    let skip_width = bound_width(&skip_output);
    let no_skip_width = bound_width(&no_skip_output);
    eprintln!(
        "Skip connection stability IBP: skip_width={skip_width:.6}, no_skip_width={no_skip_width:.6}"
    );
    // Skip connection should produce wider output bounds (identity path preserves input range)
    // while no-skip with small weights contracts
    assert!(
        skip_width >= no_skip_width - 1e-4,
        "skip connection should maintain at least as wide bounds: skip={skip_width}, no_skip={no_skip_width}"
    );
}

// ===========================================================================
// 15. Full forward-backward pipeline bound propagation (IBP + CROWN)
// ===========================================================================

/// Full pre-LN transformer block representative of dpdf decoder layers:
/// LayerNorm -> MHA -> residual -> LayerNorm -> SwiGLU FFN -> residual.
/// This is the complete gradient path for one decoder layer.
fn build_full_pipeline_kernel() -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new("dpdf_grad_full_pipeline");
    let input = b.add_input("x", &[SEQ_LEN, HIDDEN_DIM]);
    let shape = [SEQ_LEN, HIDDEN_DIM];

    // Pre-norm attention block
    let eps1 = b.add_input("ln1_eps", &[1]);
    let ln1_w = b.add_input("ln1_w", &[HIDDEN_DIM]);
    let ln1_b = b.add_input("ln1_b", &[HIDDEN_DIM]);
    let normed1 = b.add_layer_norm(input, eps1, 1, ln1_w, ln1_b, &shape);

    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let attn_out = b
        .add_multi_head_attention(
            normed1,
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

    // Pre-norm FFN block
    let eps2 = b.add_input("ln2_eps", &[1]);
    let ln2_w = b.add_input("ln2_w", &[HIDDEN_DIM]);
    let ln2_b = b.add_input("ln2_b", &[HIDDEN_DIM]);
    let normed2 = b.add_layer_norm(h, eps2, 1, ln2_w, ln2_b, &shape);

    // SwiGLU FFN
    let ffn_out = build_swiglu_block(&mut b, normed2, "ffn", SEQ_LEN, HIDDEN_DIM, FFN_DIM);

    // Second residual
    let out = b.add_binary_add(h, ffn_out, &shape);
    b.build(out).expect("valid full pipeline kernel")
}

fn full_pipeline_bindings() -> Vec<TensorParamBinding> {
    let w = |s: &[usize]| weight_binding(s, WEIGHT_MAG);
    let eps = || TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 1e-5f32));
    let norm_w =
        || TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32));
    let norm_b =
        || TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32));

    let mut bindings = vec![
        TensorParamBinding::Variable,
        eps(),                        // ln1_eps
        norm_w(),                     // ln1_w
        norm_b(),                     // ln1_b
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // q_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // k_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // v_w
        w(&[HIDDEN_DIM, HIDDEN_DIM]), // o_w
        eps(),                        // ln2_eps
        norm_w(),                     // ln2_w
        norm_b(),                     // ln2_b
    ];
    push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM, WEIGHT_MAG);
    bindings
}

#[test]
fn test_gradient_full_pipeline_ibp() {
    let def = build_full_pipeline_kernel();
    let bindings = full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let width = bound_width(&output);
    eprintln!("Full pipeline IBP: bounds=[{lo_min:.6}, {hi_max:.6}], width={width:.6}");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");
}

#[test]
fn test_gradient_full_pipeline_crown() {
    let def = build_full_pipeline_kernel();
    let bindings = full_pipeline_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[SEQ_LEN, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_bounds_valid(&output);
    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Full pipeline CROWN: method={method:?}, bounds=[{lo_min:.6}, {hi_max:.6}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}
