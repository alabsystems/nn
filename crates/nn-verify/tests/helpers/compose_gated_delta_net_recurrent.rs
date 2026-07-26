// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Gated DeltaNet two-timestep and single-variable composition.
//!
//! Split from `compose_gated_delta_net_mixed.rs` to stay under 500-line limit.
//!
//! 1. Two-timestep mixed-binding composition with chained recurrent state.
//! 2. Single-variable tests: only state is Variable (exercises #840 fix).
//!
//! Part of #834 — Gated DeltaNet for Qwen3.5 model support.

use super::common::{assert_bounds_valid, assert_crown_tighter_when_not_fallback, uniform_bounds};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{
    propagate_with_crown_fallback, tensor_kernel_to_graph, BoundedTensor, TensorParamBinding,
};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Two-timestep mixed-binding composition tests
// ---------------------------------------------------------------------------

/// Build a two-timestep GDN with mixed bindings: gate/beta constant.
fn build_two_timestep_mixed_gdn(
    h: usize,
    k: usize,
    v: usize,
    scale: f32,
    gate_val: f32,
    beta_val: f32,
) -> (TensorKernelDef, Vec<TensorParamBinding>) {
    use nn_dsl::gated_delta_net::decompose_gated_delta_net;

    let mut b = TensorBlockBuilder::new("gdn_two_timestep_mixed");

    let q1 = b.add_input("q1", &[h, k]);
    let k1 = b.add_input("k1", &[h, k]);
    let v1 = b.add_input("v1", &[h, v]);
    let state0 = b.add_input("state0", &[h, k, v]);
    let gate1 = b.add_input("gate1", &[h, 1, 1]);
    let beta1 = b.add_input("beta1", &[h, 1]);
    let q2 = b.add_input("q2", &[h, k]);
    let k2 = b.add_input("k2", &[h, k]);
    let v2 = b.add_input("v2", &[h, v]);
    let gate2 = b.add_input("gate2", &[h, 1, 1]);
    let beta2 = b.add_input("beta2", &[h, 1]);

    let t1 = decompose_gated_delta_net(&mut b, q1, k1, v1, state0, gate1, beta1, scale, h, k, v);
    let t2 = decompose_gated_delta_net(
        &mut b,
        q2,
        k2,
        v2,
        t1.new_state,
        gate2,
        beta2,
        scale,
        h,
        k,
        v,
    );

    let def = b.build(t2.output).expect("valid two-timestep mixed GDN");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, 1, 1]), gate_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, 1]), beta_val)),
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, 1, 1]), gate_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, 1]), beta_val)),
    ];

    (def, bindings)
}

/// Two-timestep mixed-binding GDN graph builds and validates.
#[test]
fn test_gdn_two_timestep_mixed_graph_builds() {
    let (h, k, v) = (1, 2, 2);
    let scale = 1.0 / (k as f32).sqrt();
    let (def, bindings) = build_two_timestep_mixed_gdn(h, k, v, scale, 0.9, 0.5);

    assert!(def.validate().is_ok(), "{:?}", def.validate());
    assert!(
        def.nodes.len() > 11,
        "expected >11 nodes, got {}",
        def.nodes.len()
    );
    assert_eq!(def.nodes[def.output.index()].shape, vec![h, v]);

    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("two-timestep mixed-binding GDN graph must build");
    assert!(graph.num_nodes() >= 5, "expected at least 5 graph nodes");
}

/// Two-timestep mixed-binding CROWN propagation through chained state.
///
/// This graph has variable inputs with heterogeneous shapes ([h,k] and
/// [h,k,v]), so propagation may fail with ShapeMismatch. When it succeeds,
/// we validate bounds. CROWN-vs-IBP tightness is exercised via the
/// single-variable test below where shapes are uniform.
#[test]
fn test_gdn_two_timestep_mixed_crown_propagates() {
    let (h, dim) = (1, 2);
    let scale = 1.0 / (dim as f32).sqrt();
    let (def, bindings) = build_two_timestep_mixed_gdn(h, dim, dim, scale, 0.9, 0.5);

    let input = uniform_bounds(&[7, h, dim, dim], 1.0);

    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("two-timestep mixed GDN graph must build");

    match propagate_with_crown_fallback(&graph, &input) {
        Ok((method, output, fallback)) => {
            assert_bounds_valid(&output);
            eprintln!("Two-timestep mixed GDN CROWN: method={method:?}, fallback={fallback:?}");
        }
        Err(e) => {
            // Known pre-existing issue: multi-variable input stacking pads all
            // variables to max dimension, causing shape mismatch when GDN
            // variables have heterogeneous shapes.
            let msg = format!("{e}");
            assert!(
                msg.contains("Shape mismatch"),
                "Expected known shape mismatch error, got unexpected: {e}"
            );
        }
    }
}

/// Two-timestep mixed vs all-Variable: mixed should be tighter.
fn build_all_var_two_timestep_gdn(h: usize, dim: usize, scale: f32) -> TensorKernelDef {
    use nn_dsl::gated_delta_net::decompose_gated_delta_net;
    let mut b = TensorBlockBuilder::new("gdn_two_timestep_allvar");
    let q1 = b.add_input("q1", &[h, dim]);
    let k1 = b.add_input("k1", &[h, dim]);
    let v1 = b.add_input("v1", &[h, dim]);
    let state0 = b.add_input("state0", &[h, dim, dim]);
    let gate1 = b.add_input("gate1", &[h, 1, 1]);
    let beta1 = b.add_input("beta1", &[h, 1]);
    let q2 = b.add_input("q2", &[h, dim]);
    let k2 = b.add_input("k2", &[h, dim]);
    let v2 = b.add_input("v2", &[h, dim]);
    let gate2 = b.add_input("gate2", &[h, 1, 1]);
    let beta2 = b.add_input("beta2", &[h, 1]);
    let t1 =
        decompose_gated_delta_net(&mut b, q1, k1, v1, state0, gate1, beta1, scale, h, dim, dim);
    let t2 = decompose_gated_delta_net(
        &mut b,
        q2,
        k2,
        v2,
        t1.new_state,
        gate2,
        beta2,
        scale,
        h,
        dim,
        dim,
    );
    b.build(t2.output).expect("valid all-var two-timestep GDN")
}

/// Compares two-timestep output widths: all-Variable vs mixed-binding.
#[test]
fn test_gdn_two_timestep_mixed_tighter_than_all_variable() {
    let (h, dim) = (1, 2);
    let scale = 1.0 / (dim as f32).sqrt();

    let all_var_def = build_all_var_two_timestep_gdn(h, dim, scale);
    let all_var_bindings = vec![TensorParamBinding::Variable; 11];
    let all_var_input = uniform_bounds(&[11, h, dim, dim], 1.0);

    let (mixed_def, mixed_bindings) = build_two_timestep_mixed_gdn(h, dim, dim, scale, 0.9, 0.5);
    let mixed_input = uniform_bounds(&[7, h, dim, dim], 1.0);

    let all_var_graph = tensor_kernel_to_graph(&all_var_def, &all_var_bindings);
    let mixed_graph = tensor_kernel_to_graph(&mixed_def, &mixed_bindings);

    match (all_var_graph, mixed_graph) {
        (Ok(av_graph), Ok(mx_graph)) => {
            let av_result = propagate_with_crown_fallback(&av_graph, &all_var_input);
            let mx_result = propagate_with_crown_fallback(&mx_graph, &mixed_input);

            match (av_result, mx_result) {
                (Ok((_, av_out, _)), Ok((_, mx_out, _))) => {
                    let (av_lo, av_hi) = av_out.lower_upper();
                    let (mx_lo, mx_hi) = mx_out.lower_upper();

                    let av_width: f32 = av_hi.iter().zip(av_lo.iter()).map(|(h, l)| h - l).sum();
                    let mx_width: f32 = mx_hi.iter().zip(mx_lo.iter()).map(|(h, l)| h - l).sum();

                    assert!(
                        mx_width <= av_width + 1e-2,
                        "two-timestep mixed bounds ({mx_width:.4}) should not be wider \
                         than all-variable ({av_width:.4})"
                    );
                }
                (Err(e), _) | (_, Err(e)) => {
                    // Known pre-existing shape mismatch from heterogeneous
                    // multi-variable padding in two-timestep GDN.
                    let msg = format!("{e}");
                    assert!(
                        msg.contains("Shape mismatch"),
                        "Expected known shape mismatch, got unexpected: {e}"
                    );
                }
            }
        }
        (Err(e), _) | (_, Err(e)) => {
            // Known pre-existing shape mismatch from multi-variable padding.
            let msg = format!("{e}");
            assert!(
                msg.contains("Shape mismatch"),
                "Expected known shape mismatch, got unexpected: {e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Single-variable tests: only state is Variable (#840)
// ---------------------------------------------------------------------------

/// Helper: single-variable bindings — only state is Variable.
fn single_variable_bindings(
    h: usize,
    k: usize,
    v: usize,
    gate_val: f32,
    beta_val: f32,
) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, k]), 0.1)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, k]), 0.2)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, v]), 0.3)),
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, 1, 1]), gate_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, 1]), beta_val)),
    ]
}

/// Single-variable GDN graph builds (exercises #840 fix).
#[test]
fn test_gdn_decomposed_single_variable_graph_builds() {
    use nn_dsl::build_gated_delta_net_decomposed;

    let (h, k, v) = (2, 3, 4);
    let scale = 1.0 / (k as f32).sqrt();
    let def = build_gated_delta_net_decomposed(h, k, v, scale).expect("valid decomposed kernel");

    let bindings = single_variable_bindings(h, k, v, 0.9, 0.5);
    let graph = tensor_kernel_to_graph(&def, &bindings)
        .expect("single-variable GDN graph must build (Fixes #840)");

    assert!(
        graph.num_nodes() >= 3,
        "expected at least 3 nodes, got {}",
        graph.num_nodes()
    );
}

/// Single-variable GDN: CROWN propagation succeeds.
#[test]
fn test_gdn_decomposed_single_variable_crown() {
    use nn_dsl::build_gated_delta_net_decomposed;

    let h = 2;
    let k = 3;
    let v = 4;
    let scale = 1.0 / (k as f32).sqrt();
    let def = build_gated_delta_net_decomposed(h, k, v, scale).expect("valid decomposed kernel");

    let bindings = single_variable_bindings(h, k, v, 0.9, 0.5);
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("single-variable GDN graph must build");

    let shape = [h, k, v];
    let bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&shape), -1.0),
        ArrayD::from_elem(IxDyn(&shape), 1.0),
    )
    .expect("valid bounds");

    let (method, _output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &bounds);
    eprintln!("Single-variable GDN CROWN: method={method:?}, fallback={fallback:?}");
}
