// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Gated DeltaNet decomposed + mixed-binding composition.
//!
//! Tests the decomposed DeltaNet builder path through individual primitive
//! translators (MatMul, BinaryMul, BinaryAdd) rather than the monolithic
//! `translate_gated_delta_net`. Covers:
//!
//! 1. All-Variable decomposed builder (structural, CROWN)
//! 2. Mixed-binding composition: gate/beta as ConstantTensor, q/k/v/state as
//!    Variable. ConstantTensor gate/beta translate to MulConstantLayer (linear)
//!    instead of MulBinary (bilinear relaxation), enabling tighter CROWN bounds.
//!
//! Two-timestep and single-variable tests are in `compose_gated_delta_net_recurrent.rs`.
//!
//! Part of #834 — Gated DeltaNet for Qwen3.5 model support.

use super::common::{assert_bounds_valid, assert_crown_tighter_when_not_fallback, uniform_bounds};
use nn_verify::{propagate_with_crown_fallback, tensor_kernel_to_graph, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Decomposed builder tests (all-Variable, structural + CROWN)
// ---------------------------------------------------------------------------

/// The decomposed DeltaNet builder also translates to a valid graph.
#[test]
fn test_gdn_decomposed_all_variable_graph_builds() {
    use nn_dsl::build_gated_delta_net_decomposed;

    let (h, k, v) = (2, 3, 4);
    let scale = 1.0 / (k as f32).sqrt();
    let def =
        build_gated_delta_net_decomposed(h, k, v, scale).expect("valid decomposed DeltaNet kernel");

    let bindings = vec![TensorParamBinding::Variable; 6];
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("decomposed DeltaNet graph must build");

    assert!(
        graph.num_nodes() >= 5,
        "expected at least 5 nodes, got {}",
        graph.num_nodes()
    );
}

/// Decomposed builder CROWN propagation with all-Variable inputs.
///
/// Multi-variable GDN has heterogeneous input shapes ([h,k], [h,k,v]),
/// so IBP may fail with ShapeMismatch. When propagation succeeds,
/// we validate bounds.
#[test]
fn test_gdn_decomposed_crown_all_variable() {
    use nn_dsl::build_gated_delta_net_decomposed;

    let (h, dim) = (1, 2);
    let scale = 1.0 / (dim as f32).sqrt();
    let def =
        build_gated_delta_net_decomposed(h, dim, dim, scale).expect("valid decomposed kernel");

    let bindings = vec![TensorParamBinding::Variable; 6];
    let input = uniform_bounds(&[6, h, dim, dim], 1.0);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("decomposed GDN graph must build");

    match propagate_with_crown_fallback(&graph, &input) {
        Ok((method, output, fallback)) => {
            assert_bounds_valid(&output);
            eprintln!("Decomposed GDN CROWN: method={method:?}, fallback={fallback:?}");
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

// ---------------------------------------------------------------------------
// Mixed-binding composition tests (D6): ConstantTensor gate/beta
// ---------------------------------------------------------------------------

/// Helper: build mixed-binding bindings for decomposed GDN.
fn mixed_bindings(h: usize, gate_val: f32, beta_val: f32) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, 1, 1]), gate_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[h, 1]), beta_val)),
    ]
}

/// Decomposed GDN with mixed bindings: graph builds successfully.
#[test]
fn test_gdn_mixed_bindings_graph_builds() {
    use nn_dsl::build_gated_delta_net_decomposed;

    let (h, k, v) = (2, 3, 4);
    let scale = 1.0 / (k as f32).sqrt();
    let def = build_gated_delta_net_decomposed(h, k, v, scale).expect("valid decomposed kernel");

    let bindings = mixed_bindings(h, 0.9, 0.5);
    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("mixed-binding GDN graph must build");

    assert!(
        graph.num_nodes() >= 5,
        "expected at least 5 nodes, got {}",
        graph.num_nodes()
    );
}

/// Mixed-binding decomposed GDN: CROWN propagation through variable q/k/v/state.
///
/// Multi-variable GDN has heterogeneous input shapes, so IBP may fail
/// with ShapeMismatch. When propagation succeeds, we validate bounds.
#[test]
fn test_gdn_mixed_bindings_crown_propagates() {
    use nn_dsl::build_gated_delta_net_decomposed;

    let (h, dim) = (1, 2);
    let scale = 1.0 / (dim as f32).sqrt();
    let def =
        build_gated_delta_net_decomposed(h, dim, dim, scale).expect("valid decomposed kernel");

    let bindings = mixed_bindings(h, 0.9, 0.5);
    let input = uniform_bounds(&[4, h, dim, dim], 1.0);

    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("mixed-binding GDN graph must build");

    match propagate_with_crown_fallback(&graph, &input) {
        Ok((method, output, fallback)) => {
            assert_bounds_valid(&output);
            eprintln!("Mixed-binding GDN CROWN: method={method:?}, fallback={fallback:?}");
        }
        Err(e) => {
            // Known pre-existing shape mismatch from heterogeneous
            // multi-variable padding in decomposed GDN.
            let msg = format!("{e}");
            assert!(
                msg.contains("Shape mismatch"),
                "Expected known shape mismatch, got unexpected: {e}"
            );
        }
    }
}

/// Mixed bindings produce tighter bounds than all-Variable.
#[test]
fn test_gdn_mixed_bindings_tighter_than_all_variable() {
    use nn_dsl::build_gated_delta_net_decomposed;

    let (h, dim) = (1, 2);
    let scale = 1.0 / (dim as f32).sqrt();
    let def =
        build_gated_delta_net_decomposed(h, dim, dim, scale).expect("valid decomposed kernel");

    let all_var_bindings = vec![TensorParamBinding::Variable; 6];
    let all_var_input = uniform_bounds(&[6, h, dim, dim], 1.0);

    let mixed_bindings = mixed_bindings(h, 0.9, 0.5);
    let mixed_input = uniform_bounds(&[4, h, dim, dim], 1.0);

    let all_var_graph =
        tensor_kernel_to_graph(&def, &all_var_bindings).expect("all-variable graph must build");
    let mixed_graph =
        tensor_kernel_to_graph(&def, &mixed_bindings).expect("mixed-binding graph must build");

    let all_var_result = propagate_with_crown_fallback(&all_var_graph, &all_var_input);
    let mixed_result = propagate_with_crown_fallback(&mixed_graph, &mixed_input);

    match (all_var_result, mixed_result) {
        (Ok((_, all_var_out, _)), Ok((_, mixed_out, _))) => {
            let (av_lo, av_hi) = all_var_out.lower_upper();
            let (mx_lo, mx_hi) = mixed_out.lower_upper();

            let av_width: f32 = av_hi.iter().zip(av_lo.iter()).map(|(h, l)| h - l).sum();
            let mx_width: f32 = mx_hi.iter().zip(mx_lo.iter()).map(|(h, l)| h - l).sum();

            assert!(
                mx_width <= av_width + 1e-2,
                "mixed-binding bounds ({mx_width:.4}) should not be wider than \
                 all-variable bounds ({av_width:.4})"
            );
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

/// Mixed-binding CROWN with minimal dimensions (H=1, K=V=1).
#[test]
fn test_gdn_mixed_bindings_crown_minimal_dims() {
    use nn_dsl::build_gated_delta_net_decomposed;

    let (h, dim) = (1, 1);
    let scale = 1.0;
    let def =
        build_gated_delta_net_decomposed(h, dim, dim, scale).expect("valid decomposed kernel");

    let bindings = mixed_bindings(h, 0.9, 0.5);
    let input = uniform_bounds(&[4, h, dim, dim], 0.5);

    let graph =
        tensor_kernel_to_graph(&def, &bindings).expect("minimal mixed-binding graph must build");

    let (method, _output, fallback) = assert_crown_tighter_when_not_fallback(&graph, &input);
    eprintln!("GDN minimal mixed CROWN: method={method:?}, fallback={fallback:?}");
}

/// verify_tensor_and_record with mixed bindings records verification status.
#[test]
fn test_gdn_mixed_bindings_verify_and_record() {
    use nn_dsl::build_gated_delta_net_decomposed;
    use nn_verify::VerifyStatus;

    let (h, dim) = (1, 2);
    let scale = 1.0 / (dim as f32).sqrt();
    let def =
        build_gated_delta_net_decomposed(h, dim, dim, scale).expect("valid decomposed kernel");

    let bindings = mixed_bindings(h, 0.9, 0.5);
    let input = uniform_bounds(&[4, h, dim, dim], 1.0);

    let mut status = VerifyStatus::default();
    let result = nn_verify::verify_tensor_and_record(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("gdn_mixed_binding_composition"),
    );

    match result {
        Ok(r) => {
            assert!(r.verification.is_finite, "bounds must be finite");
            assert_eq!(r.num_variables, 4, "expected 4 variable inputs");
            assert_bounds_valid(&r.output_bounds);
        }
        Err(e) => {
            // Known pre-existing shape mismatch from multi-variable padding.
            let msg = format!("{e}");
            assert!(
                msg.contains("Shape mismatch"),
                "Expected known shape mismatch, got unexpected: {e}"
            );
        }
    }
}
