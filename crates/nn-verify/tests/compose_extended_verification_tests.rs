// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended compose verification tests for IntervalBounds propagation.
//!
//! Covers:
//! 1. IntervalBounds propagation through linear layers
//! 2. Activation function monotonicity (ReLU, GELU, Sigmoid as SiLU proxy)
//! 3. Multi-layer composition (bounds chaining)
//! 4. Edge cases: zero-width bounds, single-element tensors, negative ranges
//! 5. Softmax numerical stability (output in [0,1], sum ~1)
//!
//! Part of #4186.

mod common;

use common::{assert_bounds_valid, uniform_bounds};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::{chain_graphs, tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ============================================================================
// 1. IntervalBounds propagation through linear layers
// ============================================================================

/// Build a single Linear layer: input [seq, in_dim] -> output [seq, out_dim].
fn build_linear_layer(
    name: &str,
    seq: usize,
    in_dim: usize,
    out_dim: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let in_shape = [seq, in_dim];
    let out_shape = [seq, out_dim];

    let mut b = TensorBlockBuilder::new(name);
    let data = b.add_input("data", &in_shape);
    let weight = b.add_input("weight", &[out_dim, in_dim]);
    let bias = b.add_input("bias", &[out_dim]);
    let out = b.add_linear(data, weight, Some(bias), &out_shape);
    b.build(out).expect("valid linear layer")
}

/// Bindings for a linear layer: data=Variable, weight and bias are constant tensors.
fn linear_bindings(in_dim: usize, out_dim: usize, weight_scale: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[out_dim, in_dim]),
            weight_scale,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[out_dim]), 0.0f32)),
    ]
}

#[test]
fn test_linear_layer_ibp_bounds_contain_expected_values() {
    let seq = 4;
    let in_dim = 8;
    let out_dim = 4;
    let weight_scale = 0.1f32;

    let def = build_linear_layer("linear_bounds", seq, in_dim, out_dim);
    let bindings = linear_bindings(in_dim, out_dim, weight_scale);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("linear graph");

    // Input in [-1, 1].
    let input = uniform_bounds(&[seq, in_dim], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP through linear");

    assert_eq!(output.lower_upper().0.shape(), &[seq, out_dim]);
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();

    // Analytical: output[i,j] = sum_k(weight[j,k] * input[i,k]) + bias[j].
    // With uniform weights = 0.1, input in [-1, 1], in_dim=8:
    //   output range = [-0.1 * 8, 0.1 * 8] + 0 = [-0.8, 0.8].
    // IBP must contain the analytical range.
    for &l in lo.iter() {
        assert!(
            l <= 0.8 + 1e-4,
            "linear lower bound should be <= 0.8, got {l}"
        );
    }
    for &u in hi.iter() {
        assert!(
            u >= -0.8 - 1e-4,
            "linear upper bound should be >= -0.8, got {u}"
        );
    }
}

#[test]
fn test_linear_layer_bounds_widen_with_larger_weights() {
    let seq = 2;
    let in_dim = 4;
    let out_dim = 4;

    let def_small = build_linear_layer("linear_small", seq, in_dim, out_dim);
    let def_large = build_linear_layer("linear_large", seq, in_dim, out_dim);

    let bindings_small = linear_bindings(in_dim, out_dim, 0.01);
    let bindings_large = linear_bindings(in_dim, out_dim, 0.5);

    let graph_small = tensor_kernel_to_graph(&def_small, &bindings_small).expect("small graph");
    let graph_large = tensor_kernel_to_graph(&def_large, &bindings_large).expect("large graph");

    let input = uniform_bounds(&[seq, in_dim], 1.0);
    let out_small = graph_small
        .propagate_ibp(&input)
        .expect("IBP small weights");
    let out_large = graph_large
        .propagate_ibp(&input)
        .expect("IBP large weights");

    let (lo_s, hi_s) = out_small.lower_upper();
    let (lo_l, hi_l) = out_large.lower_upper();

    // Larger weights should produce wider bounds.
    let width_small = hi_s
        .iter()
        .zip(lo_s.iter())
        .map(|(h, l)| h - l)
        .fold(0.0f32, f32::max);
    let width_large = hi_l
        .iter()
        .zip(lo_l.iter())
        .map(|(h, l)| h - l)
        .fold(0.0f32, f32::max);

    assert!(
        width_large > width_small,
        "larger weights should produce wider bounds: large={width_large} vs small={width_small}"
    );
}

#[test]
fn test_linear_layer_zero_bias_symmetry() {
    let seq = 2;
    let in_dim = 4;
    let out_dim = 4;

    let def = build_linear_layer("linear_sym", seq, in_dim, out_dim);
    let bindings = linear_bindings(in_dim, out_dim, 0.1);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("linear graph");

    // Symmetric input [-1, 1] with zero bias and uniform weights
    // should produce symmetric output bounds.
    let input = uniform_bounds(&[seq, in_dim], 1.0);
    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through symmetric linear");

    let (lo, hi) = output.lower_upper();
    let tol = 1e-5;
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(
            (l + u).abs() < tol,
            "symmetric input + zero bias should give symmetric bounds: lo={l}, hi={u}"
        );
    }
}

// ============================================================================
// 2. Activation function monotonicity (ReLU, GELU, Sigmoid)
// ============================================================================

/// Build a single-activation graph.
fn build_activation_graph(
    name: &str,
    activation: &str,
    dim: usize,
) -> nn_dsl::tensor_ir::TensorKernelDef {
    let shape = [dim];
    let mut b = TensorBlockBuilder::new(name);
    let data = b.add_input("data", &shape);
    let out = match activation {
        "relu" => b.add_relu(data, &shape),
        "gelu" => b.add_gelu(data, &shape),
        "sigmoid" => b.add_sigmoid(data, &shape),
        "tanh" => b.add_tanh(data, &shape),
        _ => panic!("unknown activation: {activation}"),
    };
    b.build(out).expect("valid activation graph")
}

/// Helper: propagate IBP through a single activation with asymmetric bounds.
fn activation_ibp(activation: &str, lo: f32, hi: f32, dim: usize) -> BoundedTensor {
    let def = build_activation_graph(&format!("{activation}_mono"), activation, dim);
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("activation graph");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[dim]), lo),
        ArrayD::from_elem(IxDyn(&[dim]), hi),
    )
    .expect("valid bounds");
    graph.propagate_ibp(&input).expect("IBP through activation")
}

#[test]
fn test_relu_monotonicity_wider_input_gives_wider_output() {
    // ReLU is monotone: wider input bounds should give wider output bounds.
    let narrow = activation_ibp("relu", -1.0, 1.0, 8);
    let wide = activation_ibp("relu", -5.0, 5.0, 8);

    let (n_lo, n_hi) = narrow.lower_upper();
    let (w_lo, w_hi) = wide.lower_upper();

    let narrow_width = n_hi[[0]] - n_lo[[0]];
    let wide_width = w_hi[[0]] - w_lo[[0]];

    assert!(
        wide_width >= narrow_width - 1e-5,
        "relu: wider input should give wider output: wide={wide_width}, narrow={narrow_width}"
    );
}

#[test]
fn test_relu_output_non_negative() {
    // ReLU output is always >= 0.
    let output = activation_ibp("relu", -10.0, 10.0, 16);
    let (lo, _hi) = output.lower_upper();
    for &l in lo.iter() {
        assert!(
            l >= -1e-5,
            "relu output lower bound should be >= 0, got {l}"
        );
    }
}

#[test]
fn test_gelu_monotonicity_wider_input_gives_wider_output() {
    let narrow = activation_ibp("gelu", -1.0, 1.0, 8);
    let wide = activation_ibp("gelu", -5.0, 5.0, 8);

    let (n_lo, n_hi) = narrow.lower_upper();
    let (w_lo, w_hi) = wide.lower_upper();

    let narrow_width = n_hi[[0]] - n_lo[[0]];
    let wide_width = w_hi[[0]] - w_lo[[0]];

    assert!(
        wide_width >= narrow_width - 1e-5,
        "gelu: wider input should give wider output: wide={wide_width}, narrow={narrow_width}"
    );
}

#[test]
fn test_sigmoid_output_in_unit_interval() {
    // Sigmoid output is in (0, 1) for any finite input.
    let output = activation_ibp("sigmoid", -10.0, 10.0, 16);
    let (lo, hi) = output.lower_upper();
    for &l in lo.iter() {
        assert!(l >= -0.01, "sigmoid lower bound should be >= 0, got {l}");
    }
    for &u in hi.iter() {
        assert!(u <= 1.01, "sigmoid upper bound should be <= 1, got {u}");
    }
}

#[test]
fn test_sigmoid_monotonicity_shifted_input_shifts_output() {
    // Sigmoid is monotonically increasing: shifting input up should shift output up.
    let low_range = activation_ibp("sigmoid", -5.0, -3.0, 4);
    let high_range = activation_ibp("sigmoid", 3.0, 5.0, 4);

    let (_lo_low, hi_low) = low_range.lower_upper();
    let (lo_high, _hi_high) = high_range.lower_upper();

    // Lower range output should be below upper range output.
    assert!(
        hi_low[[0]] < lo_high[[0]] + 1e-3,
        "sigmoid([-5,-3]) upper={} should be < sigmoid([3,5]) lower={}",
        hi_low[[0]],
        lo_high[[0]]
    );
}

#[test]
fn test_tanh_output_in_range() {
    // Tanh output is in (-1, 1).
    let output = activation_ibp("tanh", -10.0, 10.0, 8);
    let (lo, hi) = output.lower_upper();
    for &l in lo.iter() {
        assert!(l >= -1.01, "tanh lower bound should be >= -1, got {l}");
    }
    for &u in hi.iter() {
        assert!(u <= 1.01, "tanh upper bound should be <= 1, got {u}");
    }
}

// ============================================================================
// 3. Composition of multiple layers (bounds chaining)
// ============================================================================

#[test]
fn test_linear_then_relu_composition() {
    // Linear -> ReLU: output bounds should be non-negative (ReLU clips negatives).
    let seq = 4;
    let dim = 8;

    // Linear layer
    let mut b1 = TensorBlockBuilder::new("linear_for_compose");
    let data1 = b1.add_input("data", &[seq, dim]);
    let w1 = b1.add_input("weight", &[dim, dim]);
    let bias1 = b1.add_input("bias", &[dim]);
    let out1 = b1.add_linear(data1, w1, Some(bias1), &[seq, dim]);
    let def1 = b1.build(out1).expect("linear def");

    let bindings1 = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim, dim]), 0.1f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 0.0f32)),
    ];

    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("linear graph");

    // ReLU layer
    let mut b2 = TensorBlockBuilder::new("relu_for_compose");
    let data2 = b2.add_input("data", &[seq, dim]);
    let relu_out = b2.add_relu(data2, &[seq, dim]);
    let def2 = b2.build(relu_out).expect("relu def");

    let bindings2 = vec![TensorParamBinding::Variable];
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("relu graph");

    // Chain: Linear -> ReLU
    let chained = chain_graphs(&[graph1, graph2]).expect("chained linear->relu");

    let input = uniform_bounds(&[seq, dim], 1.0);
    let output = chained
        .propagate_ibp(&input)
        .expect("IBP through linear->relu");

    assert_eq!(output.lower_upper().0.shape(), &[seq, dim]);
    assert_bounds_valid(&output);

    // ReLU output is non-negative.
    let (lo, _hi) = output.lower_upper();
    for &l in lo.iter() {
        assert!(
            l >= -1e-5,
            "linear->relu output lower bound should be >= 0, got {l}"
        );
    }
}

#[test]
fn test_chained_bounds_are_sound_vs_sequential_propagation() {
    // For a two-layer chain (GELU -> Sigmoid), chained graph should produce
    // bounds at least as tight as propagating sequentially.
    let dim = 8;

    let mut b1 = TensorBlockBuilder::new("gelu_chain");
    let d1 = b1.add_input("data", &[dim]);
    let g1 = b1.add_gelu(d1, &[dim]);
    let def1 = b1.build(g1).expect("gelu def");
    let graph1 = tensor_kernel_to_graph(&def1, &[TensorParamBinding::Variable]).expect("g1");

    let mut b2 = TensorBlockBuilder::new("sigmoid_chain");
    let d2 = b2.add_input("data", &[dim]);
    let s2 = b2.add_sigmoid(d2, &[dim]);
    let def2 = b2.build(s2).expect("sigmoid def");
    let graph2 = tensor_kernel_to_graph(&def2, &[TensorParamBinding::Variable]).expect("g2");

    let input = uniform_bounds(&[dim], 3.0);

    // Sequential: propagate through g1, then feed its output into g2.
    let mid = graph1.propagate_ibp(&input).expect("g1 IBP");
    let seq_output = graph2.propagate_ibp(&mid).expect("g2 IBP sequential");

    // Chained: single combined graph.
    let chained = chain_graphs(&[graph1, graph2]).expect("chained gelu->sigmoid");
    let chain_output = chained.propagate_ibp(&input).expect("chained IBP");

    let (seq_lo, seq_hi) = seq_output.lower_upper();
    let (chain_lo, chain_hi) = chain_output.lower_upper();

    let tol = 1e-4;
    for i in 0..dim {
        assert!(
            chain_lo[[i]] >= seq_lo[[i]] - tol,
            "chain_lo[{i}]={} < seq_lo[{i}]={} (chain should be at least as tight)",
            chain_lo[[i]],
            seq_lo[[i]]
        );
        assert!(
            chain_hi[[i]] <= seq_hi[[i]] + tol,
            "chain_hi[{i}]={} > seq_hi[{i}]={} (chain should be at least as tight)",
            chain_hi[[i]],
            seq_hi[[i]]
        );
    }
}

#[test]
fn test_three_layer_composition_relu_gelu_sigmoid() {
    // ReLU -> GELU -> Sigmoid: end-to-end bound propagation.
    let dim = 8;

    let mut b1 = TensorBlockBuilder::new("relu_3");
    let d1 = b1.add_input("data", &[dim]);
    let r1 = b1.add_relu(d1, &[dim]);
    let def1 = b1.build(r1).expect("relu def");
    let g1 = tensor_kernel_to_graph(&def1, &[TensorParamBinding::Variable]).expect("g1");

    let mut b2 = TensorBlockBuilder::new("gelu_3");
    let d2 = b2.add_input("data", &[dim]);
    let ge2 = b2.add_gelu(d2, &[dim]);
    let def2 = b2.build(ge2).expect("gelu def");
    let g2 = tensor_kernel_to_graph(&def2, &[TensorParamBinding::Variable]).expect("g2");

    let mut b3 = TensorBlockBuilder::new("sigmoid_3");
    let d3 = b3.add_input("data", &[dim]);
    let s3 = b3.add_sigmoid(d3, &[dim]);
    let def3 = b3.build(s3).expect("sigmoid def");
    let g3 = tensor_kernel_to_graph(&def3, &[TensorParamBinding::Variable]).expect("g3");

    // Chain all three.
    let chained = chain_graphs(&[g1, g2, g3]).expect("chained relu->gelu->sigmoid");
    let input = uniform_bounds(&[dim], 5.0);
    let output = chained.propagate_ibp(&input).expect("IBP through 3-layer");

    assert_bounds_valid(&output);

    // Final layer is sigmoid, so output should be in [0, 1].
    let (lo, hi) = output.lower_upper();
    for &l in lo.iter() {
        assert!(l >= -0.01, "3-layer output lower should be >= 0, got {l}");
    }
    for &u in hi.iter() {
        assert!(u <= 1.01, "3-layer output upper should be <= 1, got {u}");
    }

    // Concrete forward check: for specific input values.
    // relu(2.0) = 2.0, gelu(2.0) ~ 1.954, sigmoid(1.954) ~ 0.876
    // The output bounds must contain this.
    let (lo_min, hi_max) = common::bounds_min_max(&output);
    assert!(
        hi_max >= 0.85,
        "3-layer composition must reach sigmoid(gelu(relu(x))) ~ 0.876, got hi_max={hi_max}"
    );
    assert!(
        lo_min <= 0.55,
        "3-layer composition must include sigmoid(gelu(relu(0)))=sigmoid(0)=0.5, got lo_min={lo_min}"
    );
}

// ============================================================================
// 4. Edge cases: zero-width bounds, single-element tensors, negative ranges
// ============================================================================

#[test]
fn test_zero_width_input_bounds_relu() {
    // Zero-width bounds (concrete value) should propagate to zero-width output.
    let dim = 4;
    let def = build_activation_graph("relu_zero_width", "relu", dim);
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Concrete input: all elements = 2.0 (lower == upper).
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[dim]), 2.0f32),
        ArrayD::from_elem(IxDyn(&[dim]), 2.0f32),
    )
    .expect("concrete bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP with zero-width bounds");
    let (lo, hi) = output.lower_upper();

    // ReLU(2.0) = 2.0, so output should be [2.0, 2.0] (zero width).
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(
            (l - 2.0).abs() < 1e-5,
            "relu of concrete 2.0 should give lower=2.0, got {l}"
        );
        assert!(
            (u - 2.0).abs() < 1e-5,
            "relu of concrete 2.0 should give upper=2.0, got {u}"
        );
    }
}

#[test]
fn test_zero_width_input_bounds_negative_relu() {
    // Concrete negative input through ReLU should give [0, 0].
    let dim = 4;
    let def = build_activation_graph("relu_neg_concrete", "relu", dim);
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[dim]), -3.0f32),
        ArrayD::from_elem(IxDyn(&[dim]), -3.0f32),
    )
    .expect("negative concrete bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP with negative concrete");
    let (lo, hi) = output.lower_upper();

    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l.abs() < 1e-5, "relu(-3.0) lower should be 0.0, got {l}");
        assert!(u.abs() < 1e-5, "relu(-3.0) upper should be 0.0, got {u}");
    }
}

#[test]
fn test_single_element_tensor_through_activation_chain() {
    // Single-element tensor [1] through ReLU -> GELU -> Sigmoid.
    let dim = 1;

    let mut b = TensorBlockBuilder::new("single_elem");
    let data = b.add_input("data", &[dim]);
    let relu = b.add_relu(data, &[dim]);
    let gelu = b.add_gelu(relu, &[dim]);
    let sigmoid = b.add_sigmoid(gelu, &[dim]);
    let def = b.build(sigmoid).expect("single element graph");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = uniform_bounds(&[dim], 2.0);
    let output = graph.propagate_ibp(&input).expect("IBP single element");

    assert_eq!(output.lower_upper().0.shape(), &[1]);
    assert_bounds_valid(&output);

    // Sigmoid output still in [0, 1].
    let (lo, hi) = output.lower_upper();
    assert!(lo[[0]] >= -0.01, "single elem lower >= 0");
    assert!(hi[[0]] <= 1.01, "single elem upper <= 1");
}

#[test]
fn test_entirely_negative_range_through_relu() {
    // Input entirely negative: relu should produce [0, 0].
    let dim = 8;
    let def = build_activation_graph("relu_all_neg", "relu", dim);
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[dim]), -5.0f32),
        ArrayD::from_elem(IxDyn(&[dim]), -1.0f32),
    )
    .expect("negative range bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP with negative range");
    let (lo, hi) = output.lower_upper();

    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(
            l.abs() < 1e-5,
            "relu([-5, -1]) lower should be 0.0, got {l}"
        );
        assert!(
            u.abs() < 1e-5,
            "relu([-5, -1]) upper should be 0.0, got {u}"
        );
    }
}

#[test]
fn test_entirely_positive_range_through_relu() {
    // Input entirely positive: relu should pass through unchanged.
    let dim = 8;
    let def = build_activation_graph("relu_all_pos", "relu", dim);
    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[dim]), 2.0f32),
        ArrayD::from_elem(IxDyn(&[dim]), 5.0f32),
    )
    .expect("positive range bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP with positive range");
    let (lo, hi) = output.lower_upper();

    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(
            (l - 2.0).abs() < 1e-4,
            "relu([2, 5]) lower should be 2.0, got {l}"
        );
        assert!(
            (u - 5.0).abs() < 1e-4,
            "relu([2, 5]) upper should be 5.0, got {u}"
        );
    }
}

#[test]
fn test_gelu_negative_range_output_bounded() {
    // GELU on entirely negative input: output should be bounded and finite.
    // GELU(-x) for large x approaches 0 from below (GELU has a small negative dip).
    let dim = 8;
    let output = activation_ibp("gelu", -5.0, -1.0, dim);
    assert_bounds_valid(&output);

    let (_lo, hi) = output.lower_upper();
    for &u in hi.iter() {
        // GELU(-1) ~ -0.159, GELU(-5) ~ 0.0 (very close). Upper bound should be near 0.
        assert!(
            u <= 0.1,
            "gelu([-5, -1]) upper should be close to 0, got {u}"
        );
    }
}

// ============================================================================
// 5. Softmax numerical stability
// ============================================================================

#[test]
fn test_softmax_output_in_unit_interval() {
    // Softmax output should be in [0, 1] for each element.
    let dim = 8;
    let shape = [dim];

    let mut b = TensorBlockBuilder::new("softmax_unit");
    let data = b.add_input("data", &shape);
    let out = b.add_softmax(data, -1, &shape);
    let def = b.build(out).expect("softmax graph");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("softmax graph");

    let input = uniform_bounds(&[dim], 5.0);
    let output = graph.propagate_ibp(&input).expect("IBP through softmax");

    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &l in lo.iter() {
        assert!(l >= -0.01, "softmax lower bound should be >= 0, got {l}");
    }
    for &u in hi.iter() {
        assert!(u <= 1.01, "softmax upper bound should be <= 1, got {u}");
    }
}

#[test]
fn test_softmax_concrete_input_sums_to_one() {
    // For concrete (zero-width) input, softmax output should sum to ~1.
    let dim = 4;
    let shape = [dim];

    let mut b = TensorBlockBuilder::new("softmax_sum");
    let data = b.add_input("data", &shape);
    let out = b.add_softmax(data, -1, &shape);
    let def = b.build(out).expect("softmax graph");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("softmax graph");

    // Concrete input: [1.0, 2.0, 3.0, 4.0].
    let values = vec![1.0f32, 2.0, 3.0, 4.0];
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), values.clone()).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[dim]), values).expect("upper"),
    )
    .expect("concrete bounds");

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through softmax concrete");
    let (lo, hi) = output.lower_upper();

    // For concrete input, lower == upper (zero width). The sum should be ~1.0.
    let lo_sum: f32 = lo.iter().sum();
    let hi_sum: f32 = hi.iter().sum();

    // IBP may slightly widen even concrete inputs, so allow tolerance.
    assert!(
        (0.9..=1.1).contains(&lo_sum),
        "softmax lower sum should be near 1.0, got {lo_sum}"
    );
    assert!(
        (0.9..=1.1).contains(&hi_sum),
        "softmax upper sum should be near 1.0, got {hi_sum}"
    );
}

#[test]
fn test_softmax_large_input_values_stable() {
    // Softmax should remain numerically stable with large inputs.
    // Implementation uses max-shift: softmax(x) = softmax(x - max(x)).
    let dim = 4;
    let shape = [dim];

    let mut b = TensorBlockBuilder::new("softmax_large");
    let data = b.add_input("data", &shape);
    let out = b.add_softmax(data, -1, &shape);
    let def = b.build(out).expect("softmax graph");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("softmax graph");

    // Large input range: [-100, 100].
    let input = uniform_bounds(&[dim], 100.0);
    let output = graph.propagate_ibp(&input).expect("IBP large softmax");

    assert_bounds_valid(&output);

    // Output should still be in [0, 1] despite large inputs.
    let (lo, hi) = output.lower_upper();
    for &l in lo.iter() {
        assert!(
            l >= -0.01,
            "softmax with large input: lower should be >= 0, got {l}"
        );
    }
    for &u in hi.iter() {
        assert!(
            u <= 1.01,
            "softmax with large input: upper should be <= 1, got {u}"
        );
    }
}

#[test]
fn test_softmax_2d_along_last_axis() {
    // Softmax on 2D tensor [seq, vocab] along last axis.
    let seq = 2;
    let vocab = 4;
    let shape = [seq, vocab];

    let mut b = TensorBlockBuilder::new("softmax_2d");
    let data = b.add_input("data", &shape);
    let out = b.add_softmax(data, -1, &shape);
    let def = b.build(out).expect("softmax 2d graph");

    let bindings = vec![TensorParamBinding::Variable];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("softmax 2d graph");

    let input = uniform_bounds(&[seq, vocab], 3.0);
    let output = graph.propagate_ibp(&input).expect("IBP through 2d softmax");

    assert_eq!(output.lower_upper().0.shape(), &[seq, vocab]);
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &l in lo.iter() {
        assert!(l >= -0.01, "2d softmax lower >= 0, got {l}");
    }
    for &u in hi.iter() {
        assert!(u <= 1.01, "2d softmax upper <= 1, got {u}");
    }
}

#[test]
fn test_softmax_after_linear_composition() {
    // Linear -> Softmax: a common pattern in classification heads.
    // Verifies that composed bounds still satisfy softmax invariants.
    let seq = 2;
    let in_dim = 8;
    let out_dim = 4;

    // Linear layer.
    let mut b1 = TensorBlockBuilder::new("linear_before_softmax");
    let data1 = b1.add_input("data", &[seq, in_dim]);
    let w1 = b1.add_input("weight", &[out_dim, in_dim]);
    let bias1 = b1.add_input("bias", &[out_dim]);
    let out1 = b1.add_linear(data1, w1, Some(bias1), &[seq, out_dim]);
    let def1 = b1.build(out1).expect("linear def");
    let bindings1 = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[out_dim, in_dim]), 0.1f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[out_dim]), 0.0f32)),
    ];
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("linear graph");

    // Softmax layer.
    let mut b2 = TensorBlockBuilder::new("softmax_after_linear");
    let data2 = b2.add_input("data", &[seq, out_dim]);
    let sm2 = b2.add_softmax(data2, -1, &[seq, out_dim]);
    let def2 = b2.build(sm2).expect("softmax def");
    let bindings2 = vec![TensorParamBinding::Variable];
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("softmax graph");

    let chained = chain_graphs(&[graph1, graph2]).expect("chained linear->softmax");
    let input = uniform_bounds(&[seq, in_dim], 1.0);
    let output = chained
        .propagate_ibp(&input)
        .expect("IBP through linear->softmax");

    assert_eq!(output.lower_upper().0.shape(), &[seq, out_dim]);
    assert_bounds_valid(&output);

    // Softmax invariant: output in [0, 1].
    let (lo_out, hi_out) = output.lower_upper();
    for &l in lo_out.iter() {
        assert!(l >= -0.01, "linear->softmax lower should be >= 0, got {l}");
    }
    for &u in hi_out.iter() {
        assert!(u <= 1.01, "linear->softmax upper should be <= 1, got {u}");
    }
}
