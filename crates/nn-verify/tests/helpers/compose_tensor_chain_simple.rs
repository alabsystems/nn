// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Simple activation chain composition tests.
//!
//! Validates that a pure-activation chain (ReLU → GELU → Sigmoid) translates
//! through `tensor_kernel_to_graph` and produces a NY `GraphNetwork`
//! where IBP and CROWN bounds propagate end-to-end.
//!
//! Part of #2039.

use super::common::{assert_bounds_valid, assert_crown_tighter_when_not_fallback, uniform_bounds};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding};

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Build a simple activation chain: ReLU → GELU → Sigmoid.
///
/// Single variable input of shape `[dim]`. No weights or parameters —
/// pure element-wise activations only.
fn build_simple_chain(dim: usize) -> nn_dsl::tensor_ir::TensorKernelDef {
    let shape = [dim];

    let mut b = TensorBlockBuilder::new("simple_chain");
    let data = b.add_input("data", &shape);
    let relu = b.add_relu(data, &shape);
    let gelu = b.add_gelu(relu, &shape);
    let sigmoid = b.add_sigmoid(gelu, &shape);

    b.build(sigmoid).expect("valid graph")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Simple activation chain builds and translates to a NY graph.
#[test]
fn test_simple_chain_graph_builds() {
    let def = build_simple_chain(16);
    assert_eq!(def.nodes.last().unwrap().shape, vec![16]);

    let bindings = vec![TensorParamBinding::Variable];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("simple chain graph");
    // ReLU + GELU + Sigmoid = at least 3 NY nodes (plus input identity).
    assert!(
        graph.num_nodes() >= 3,
        "simple chain needs >= 3 NY nodes, got {}",
        graph.num_nodes()
    );
}

/// IBP bounds propagate through the simple activation chain.
///
/// Input in [-10, 10] — all activations are well-behaved in this range.
#[test]
fn test_simple_chain_ibp_propagates() {
    let def = build_simple_chain(16);

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[16], 10.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through simple chain");

    assert_eq!(output.lower_upper().0.shape(), &[16]);
    assert_bounds_valid(&output);

    // Sigmoid output is in [0, 1] for any input.
    let (lower, upper) = output.lower_upper();
    for &lo in lower.iter() {
        assert!(lo >= -0.01, "sigmoid lower bound should be >= 0 (got {lo})");
    }
    for &hi in upper.iter() {
        assert!(hi <= 1.01, "sigmoid upper bound should be <= 1 (got {hi})");
    }
}

/// CROWN propagation through the simple activation chain.
///
/// When CROWN succeeds (no IBP fallback), asserts bounds are tighter than IBP.
#[test]
fn test_simple_chain_crown_propagates() {
    let def = build_simple_chain(8);

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[8], 10.0);

    let (_method, output, _fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);

    assert_eq!(output.lower_upper().0.shape(), &[8]);
    assert_bounds_valid(&output);
}

/// Two-dimensional activation chain preserves shape through all layers.
#[test]
fn test_simple_chain_2d_shape_preserved() {
    let shape = [4, 8];
    let mut b = TensorBlockBuilder::new("chain_2d");
    let data = b.add_input("data", &shape);
    let relu = b.add_relu(data, &shape);
    let gelu = b.add_gelu(relu, &shape);
    let sigmoid = b.add_sigmoid(gelu, &shape);

    let def = b.build(sigmoid).expect("valid 2d graph");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("2d graph");

    let input = uniform_bounds(&[4, 8], 10.0);
    let output = graph.propagate_ibp(&input).expect("IBP through 2d chain");

    assert_eq!(output.lower_upper().0.shape(), &[4, 8]);
    assert_bounds_valid(&output);
}

// ---------------------------------------------------------------------------
// chain_graphs soundness tests (#2592)
// ---------------------------------------------------------------------------

/// chain_graphs concrete soundness: forward pass within chained bounds.
///
/// Builds two separate single-activation graphs (ReLU, Sigmoid), chains them
/// via `chain_graphs`, propagates IBP, and verifies that a concrete forward
/// pass (ReLU then Sigmoid) falls within the IBP output bounds.
///
/// This is the critical soundness test: chain_graphs must produce a combined
/// graph whose bounds are a valid over-approximation of the sequential
/// composition of the original graphs.
#[test]
fn test_chain_graphs_ibp_soundness_concrete() {
    use nn_verify::chain_graphs;

    let dim = 8;

    // Graph 1: ReLU
    let mut b1 = TensorBlockBuilder::new("relu_graph");
    let d1 = b1.add_input("data", &[dim]);
    let r1 = b1.add_relu(d1, &[dim]);
    let def1 = b1.build(r1).expect("relu graph");
    let g1 = tensor_kernel_to_graph(&def1, &[TensorParamBinding::Variable]).expect("g1");

    // Graph 2: Sigmoid
    let mut b2 = TensorBlockBuilder::new("sigmoid_graph");
    let d2 = b2.add_input("data", &[dim]);
    let s2 = b2.add_sigmoid(d2, &[dim]);
    let def2 = b2.build(s2).expect("sigmoid graph");
    let g2 = tensor_kernel_to_graph(&def2, &[TensorParamBinding::Variable]).expect("g2");

    // Chain: ReLU → Sigmoid
    let chained = chain_graphs(&[g1, g2]).expect("chained graph");

    let input = uniform_bounds(&[dim], 2.0);
    let output = chained
        .propagate_ibp(&input)
        .expect("IBP through chained graph");
    let (lo, hi) = output.lower_upper();
    assert_bounds_valid(&output);

    // Concrete forward: pick test points within input bounds [-2, 2].
    let test_inputs: Vec<f32> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
    for (i, &x) in test_inputs.iter().enumerate() {
        let relu_out = x.max(0.0);
        let sigmoid_out = 1.0 / (1.0 + (-relu_out).exp());

        assert!(
            lo[[i]] <= sigmoid_out + 1e-5,
            "chain soundness: lo[{i}]={} > forward={sigmoid_out} (input={x})",
            lo[[i]]
        );
        assert!(
            hi[[i]] >= sigmoid_out - 1e-5,
            "chain soundness: hi[{i}]={} < forward={sigmoid_out} (input={x})",
            hi[[i]]
        );
    }

    // Sigmoid(ReLU(x)) for x in [-2,2]: output in [0.5, sigmoid(2)] ≈ [0.5, 0.881].
    // For x < 0: ReLU=0, sigmoid(0)=0.5.
    // IBP bounds should capture this.
    let min_lo = lo.iter().copied().fold(f32::INFINITY, f32::min);
    let max_hi = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        min_lo <= 0.5 + 0.01,
        "chain bounds must include sigmoid(0)=0.5"
    );
    assert!(
        max_hi >= 0.88 - 0.01,
        "chain bounds must include sigmoid(2)≈0.881"
    );
}

/// chain_graphs vs sequential: chained bounds are at least as tight as sequential.
///
/// Propagates IBP through two separate graphs sequentially (output of g1 feeds g2),
/// then through the chained graph. The chained graph has full cross-layer visibility,
/// so its bounds should be at least as tight as the sequential composition.
#[test]
fn test_chain_graphs_tighter_than_sequential() {
    use nn_verify::chain_graphs;

    let dim = 16;

    // Graph 1: ReLU
    let mut b1 = TensorBlockBuilder::new("relu_seq");
    let d1 = b1.add_input("data", &[dim]);
    let r1 = b1.add_relu(d1, &[dim]);
    let def1 = b1.build(r1).expect("relu");
    let g1 = tensor_kernel_to_graph(&def1, &[TensorParamBinding::Variable]).expect("g1");

    // Graph 2: Sigmoid
    let mut b2 = TensorBlockBuilder::new("sigmoid_seq");
    let d2 = b2.add_input("data", &[dim]);
    let s2 = b2.add_sigmoid(d2, &[dim]);
    let def2 = b2.build(s2).expect("sigmoid");
    let g2 = tensor_kernel_to_graph(&def2, &[TensorParamBinding::Variable]).expect("g2");

    let input = uniform_bounds(&[dim], 2.0);

    // Sequential: g1 then g2 with output→input chaining.
    let mid = g1.propagate_ibp(&input).expect("g1 IBP");
    let seq_output = g2.propagate_ibp(&mid).expect("g2 IBP sequential");
    let (seq_lo, seq_hi) = seq_output.lower_upper();

    // Chained: single combined graph.
    let chained = chain_graphs(&[g1, g2]).expect("chained");
    let chain_output = chained.propagate_ibp(&input).expect("chained IBP");
    let (chain_lo, chain_hi) = chain_output.lower_upper();

    // Chained bounds must be no wider than sequential bounds (within fp tolerance).
    let tol = 1e-4;
    for i in 0..dim {
        assert!(
            chain_lo[[i]] >= seq_lo[[i]] - tol,
            "chain_lo[{i}]={} < seq_lo[{i}]={} (chain should be tighter)",
            chain_lo[[i]],
            seq_lo[[i]]
        );
        assert!(
            chain_hi[[i]] <= seq_hi[[i]] + tol,
            "chain_hi[{i}]={} > seq_hi[{i}]={} (chain should be tighter)",
            chain_hi[[i]],
            seq_hi[[i]]
        );
    }
}

/// chain_graphs CROWN soundness: CROWN on chained element-wise graph.
///
/// Chains ReLU → GELU on [-5, 5]. For element-wise-only chains, CROWN and IBP
/// produce identical bounds (ratio=1.0) because there are no dimension-mixing
/// layers (Linear/Conv) to create cross-neuron correlations. CROWN tightening
/// requires Linear layers — see `test_kokoro_prenorm_crown_he_scaled` in
/// `compose_kokoro_layerwise_grouped.rs` for the AC2 proof with Linear+ReLU.
///
/// This test validates that chain_graphs produces a valid GraphNetwork that
/// CROWN can successfully propagate through (soundness, not tightening).
#[test]
fn test_chain_graphs_crown_soundness() {
    use nn_verify::chain_graphs;

    let dim = 8;

    // Graph 1: ReLU
    let mut b1 = TensorBlockBuilder::new("relu_crown");
    let d1 = b1.add_input("data", &[dim]);
    let r1 = b1.add_relu(d1, &[dim]);
    let def1 = b1.build(r1).expect("relu graph");
    let g1 = tensor_kernel_to_graph(&def1, &[TensorParamBinding::Variable]).expect("g1");

    // Graph 2: GELU
    let mut b2 = TensorBlockBuilder::new("gelu_crown");
    let d2 = b2.add_input("data", &[dim]);
    let ge2 = b2.add_gelu(d2, &[dim]);
    let def2 = b2.build(ge2).expect("gelu graph");
    let g2 = tensor_kernel_to_graph(&def2, &[TensorParamBinding::Variable]).expect("g2");

    // Chain: ReLU → GELU
    let chained = chain_graphs(&[g1, g2]).expect("chained ReLU→GELU");

    // Wide input so activations span non-linear regime.
    let input = uniform_bounds(&[dim], 5.0);

    // CROWN succeeds on the chained graph and is sound (no wider than IBP).
    let (method, output, _fallback) = assert_crown_tighter_when_not_fallback(&chained, &input);
    assert_bounds_valid(&output);

    // Report the ratio: expected 1.0 for element-wise-only chains.
    let ibp_output = chained.propagate_ibp(&input).expect("IBP baseline");
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
    let (crown_lo, crown_hi) = output.lower_upper();
    let ibp_width: f32 = ibp_hi
        .iter()
        .zip(ibp_lo.iter())
        .map(|(h, l)| h - l)
        .fold(0.0f32, f32::max);
    let crown_width: f32 = crown_hi
        .iter()
        .zip(crown_lo.iter())
        .map(|(h, l)| h - l)
        .fold(0.0f32, f32::max);

    eprintln!("=== chain_graphs CROWN soundness (ReLU→GELU, element-wise) ===");
    eprintln!("  IBP max width:   {ibp_width:.6}");
    eprintln!("  CROWN max width: {crown_width:.6} (method: {method:?})");
    if crown_width > 0.0 {
        eprintln!("  IBP/CROWN ratio: {:.4}", ibp_width / crown_width);
    }
    // Element-wise chains: CROWN=IBP expected (no cross-dimension mixing).
}
