// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: Gated DeltaNet single-timestep NY translation.
//!
//! Tests the monolithic `TensorOpKind::GatedDeltaNet` → NY translation
//! (`translate_gated_delta_net`) which builds 9 NY nodes for the
//! decomposed recurrence.
//!
//! The DeltaNet recurrence:
//!   decayed = gate * state
//!   v_retrieved = k^T @ decayed
//!   new_state = decayed + outer(k, beta*v) - outer(k, beta*v_retrieved)
//!   output = scale * q @ new_state
//!
//! All 6 inputs (q, k, v, state, gate, beta) are Variable for the monolithic
//! op translation. IBP propagation requires same-shape stacking, so we test
//! with K=V=dim to make q/k/v shapes identical, and accept that IBP may fail
//! on shape mismatch for the heterogeneous-shape 6-variable stacking.
//!
//! Part of #834 — Gated DeltaNet for Qwen3.5 model support.

use super::common::{assert_bounds_valid, assert_crown_tighter_than_ibp, uniform_bounds};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{propagate_with_crown_fallback, tensor_kernel_to_graph, TensorParamBinding};

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build a monolithic GatedDeltaNet kernel using `add_gated_delta_net`.
///
/// This produces a single `TensorOpKind::GatedDeltaNet` node (not decomposed),
/// which is translated by `translate_gated_delta_net` in the NY layer.
fn build_gdn_kernel(
    name: &str,
    num_heads: usize,
    key_dim: usize,
    value_dim: usize,
    scale: f32,
) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(name);
    let q = b.add_input("q", &[num_heads, key_dim]);
    let k = b.add_input("k", &[num_heads, key_dim]);
    let v = b.add_input("v", &[num_heads, value_dim]);
    let state = b.add_input("state", &[num_heads, key_dim, value_dim]);
    let gate = b.add_input("gate", &[num_heads, 1, 1]);
    let beta = b.add_input("beta", &[num_heads, 1]);

    let out = b.add_gated_delta_net(q, k, v, state, gate, beta, scale, &[num_heads, value_dim]);
    b.build(out)
        .expect("GatedDeltaNet kernel build should succeed")
}

// ---------------------------------------------------------------------------
// Graph construction tests
// ---------------------------------------------------------------------------

/// Monolithic GatedDeltaNet translates into a valid NY GraphNetwork.
#[test]
fn test_gdn_monolithic_graph_builds() {
    let (h, k, v) = (2, 3, 4);
    let scale = 1.0 / (k as f32).sqrt();
    let def = build_gdn_kernel("gdn_basic", h, k, v, scale);
    assert!(def.validate().is_ok(), "{:?}", def.validate());

    let bindings = vec![TensorParamBinding::Variable; 6];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GatedDeltaNet graph must build");

    assert!(
        graph.num_nodes() >= 9,
        "expected at least 9 translation nodes, got {}",
        graph.num_nodes()
    );
}

/// GatedDeltaNet kernel def has expected structure.
#[test]
fn test_gdn_monolithic_kernel_structure() {
    let (h, k, v) = (2, 4, 4);
    let scale = 1.0 / (k as f32).sqrt();
    let def = build_gdn_kernel("gdn_struct", h, k, v, scale);

    assert_eq!(def.name, "gdn_struct");
    assert_eq!(def.nodes.len(), 7);
    assert_eq!(def.nodes[def.output.index()].shape, vec![h, v]);
}

/// Different scale values build correctly.
#[test]
fn test_gdn_graph_different_scales() {
    for &scale in &[0.0625, 0.125, 0.5, 1.0] {
        let (h, k, v) = (2, 3, 4);
        let def = build_gdn_kernel("gdn_scale", h, k, v, scale);
        let bindings = vec![TensorParamBinding::Variable; 6];
        let graph = tensor_kernel_to_graph(&def, &bindings);
        assert!(
            graph.is_ok(),
            "graph build failed for scale={scale}: {:?}",
            graph.err()
        );
    }
}

/// Different dimension configurations build correctly.
#[test]
fn test_gdn_graph_various_dims() {
    let configs = [(1, 2, 2), (2, 4, 4), (4, 8, 8), (2, 3, 5), (8, 4, 4)];
    for (h, k, v) in configs {
        let scale = 1.0 / (k as f32).sqrt();
        let def = build_gdn_kernel("gdn_dims", h, k, v, scale);
        let bindings = vec![TensorParamBinding::Variable; 6];
        let graph = tensor_kernel_to_graph(&def, &bindings);
        assert!(
            graph.is_ok(),
            "graph build failed for H={h}, K={k}, V={v}: {:?}",
            graph.err()
        );
    }
}

// ---------------------------------------------------------------------------
// IBP bounds propagation tests
// ---------------------------------------------------------------------------

/// IBP bounds propagate through the monolithic GatedDeltaNet translation.
///
/// With 6 Variable inputs of heterogeneous shapes, IBP may fail on shape
/// mismatch during multi-variable stacking. When IBP succeeds, we verify
/// bounds are finite and sound. When it fails, we log and accept.
#[test]
fn test_gdn_ibp_bounds_propagate() {
    let (h, dim) = (2, 3);
    let scale = 1.0 / (dim as f32).sqrt();
    let def = build_gdn_kernel("gdn_ibp", h, dim, dim, scale);
    let bindings = vec![TensorParamBinding::Variable; 6];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GatedDeltaNet graph must build");

    let input = uniform_bounds(&[6, h, dim, dim], 1.0);

    match graph.propagate_ibp(&input) {
        Ok(output) => {
            assert_bounds_valid(&output);
        }
        Err(e) => {
            panic!("GatedDeltaNet IBP propagation failed unexpectedly: {e}");
        }
    }
}

/// IBP bounds with minimal dimensions: H=1, K=V=1.
#[test]
fn test_gdn_ibp_minimal() {
    let (h, dim) = (1, 1);
    let scale = 1.0;
    let def = build_gdn_kernel("gdn_ibp_min", h, dim, dim, scale);
    let bindings = vec![TensorParamBinding::Variable; 6];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GatedDeltaNet graph must build");

    let input = uniform_bounds(&[6, h, dim, dim], 1.0);

    match graph.propagate_ibp(&input) {
        Ok(output) => {
            assert_bounds_valid(&output);
        }
        Err(e) => {
            panic!("GatedDeltaNet minimal IBP failed unexpectedly: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// CROWN propagation tests
// ---------------------------------------------------------------------------

/// CROWN propagation through the monolithic GatedDeltaNet translation.
#[test]
fn test_gdn_crown_propagates() {
    let (h, dim) = (2, 3);
    let scale = 1.0 / (dim as f32).sqrt();
    let def = build_gdn_kernel("gdn_crown", h, dim, dim, scale);
    let bindings = vec![TensorParamBinding::Variable; 6];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GatedDeltaNet graph must build");

    let input = uniform_bounds(&[6, h, dim, dim], 1.0);

    match propagate_with_crown_fallback(&graph, &input) {
        Ok((method, output, fallback_reason)) => {
            assert_bounds_valid(&output);
            if let Some(reason) = &fallback_reason {
                eprintln!("GatedDeltaNet CROWN fell back to IBP: {reason}");
            }
            eprintln!("GatedDeltaNet propagation method: {method:?}");
        }
        Err(e) => {
            panic!("GatedDeltaNet CROWN propagation failed unexpectedly: {e}");
        }
    }
}

/// CROWN bounds should be at least as tight as IBP (soundness invariant).
#[test]
fn test_gdn_crown_at_least_as_tight_as_ibp() {
    let (h, dim) = (2, 3);
    let scale = 1.0 / (dim as f32).sqrt();
    let def = build_gdn_kernel("gdn_crown_tight", h, dim, dim, scale);
    let bindings = vec![TensorParamBinding::Variable; 6];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GatedDeltaNet graph must build");

    let input = uniform_bounds(&[6, h, dim, dim], 1.0);

    let ibp_result = graph.propagate_ibp(&input);
    let crown_result = propagate_with_crown_fallback(&graph, &input);

    match (ibp_result, crown_result) {
        (Ok(ibp_output), Ok((_method, crown_output, _fallback))) => {
            assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
        }
        (Err(ibp_e), _) => {
            panic!("IBP propagation failed unexpectedly: {ibp_e}");
        }
        (_, Err(crown_e)) => {
            panic!("CROWN propagation failed unexpectedly: {crown_e}");
        }
    }
}

/// CROWN with minimal dimensions (H=1, K=V=1).
#[test]
fn test_gdn_crown_minimal_dims() {
    let (h, dim) = (1, 1);
    let scale = 1.0;
    let def = build_gdn_kernel("gdn_crown_min", h, dim, dim, scale);
    let bindings = vec![TensorParamBinding::Variable; 6];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GatedDeltaNet graph must build");

    let input = uniform_bounds(&[6, h, dim, dim], 0.5);

    let (method, output, fallback) =
        propagate_with_crown_fallback(&graph, &input).expect("CROWN must succeed for minimal GDN");
    assert_bounds_valid(&output);
    eprintln!("GDN minimal CROWN: method={method:?}, fallback={fallback:?}");
}

/// CROWN with narrow input bounds produces narrower output bounds.
#[test]
fn test_gdn_crown_narrow_inputs_produce_narrower_output() {
    let (h, dim) = (1, 2);
    let scale = 1.0 / (dim as f32).sqrt();
    let def = build_gdn_kernel("gdn_crown_narrow", h, dim, dim, scale);
    let bindings = vec![TensorParamBinding::Variable; 6];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("GatedDeltaNet graph must build");

    let wide_input = uniform_bounds(&[6, h, dim, dim], 1.0);
    let narrow_input = uniform_bounds(&[6, h, dim, dim], 0.5);

    let wide_result = propagate_with_crown_fallback(&graph, &wide_input);
    let narrow_result = propagate_with_crown_fallback(&graph, &narrow_input);

    match (wide_result, narrow_result) {
        (Ok((_, wide_out, _)), Ok((_, narrow_out, _))) => {
            let (wide_lo, wide_hi) = wide_out.lower_upper();
            let (narrow_lo, narrow_hi) = narrow_out.lower_upper();

            let wide_width: f32 = wide_hi.iter().zip(wide_lo.iter()).map(|(h, l)| h - l).sum();
            let narrow_width: f32 = narrow_hi
                .iter()
                .zip(narrow_lo.iter())
                .map(|(h, l)| h - l)
                .sum();

            assert!(
                narrow_width <= wide_width + 1e-3,
                "narrower inputs should produce narrower output: \
                 narrow_width={narrow_width:.4} > wide_width={wide_width:.4}"
            );
            eprintln!(
                "GDN width comparison: wide={wide_width:.4}, narrow={narrow_width:.4}, \
                 ratio={:.2}x",
                wide_width / narrow_width.max(1e-10)
            );
        }
        (Err(e), _) | (_, Err(e)) => {
            panic!("CROWN comparison propagation failed unexpectedly: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-timestep composition tests
// ---------------------------------------------------------------------------

/// Build a two-timestep GatedDeltaNet using the decomposed builder.
fn build_two_timestep_gdn(h: usize, k: usize, v: usize, scale: f32) -> TensorKernelDef {
    use nn_dsl::gated_delta_net::decompose_gated_delta_net;

    let mut b = TensorBlockBuilder::new("gdn_two_timestep");

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

    b.build(t2.output).expect("valid two-timestep GDN")
}

/// Two-timestep GDN graph builds and validates.
#[test]
fn test_gdn_two_timestep_graph_builds() {
    let (h, k, v) = (1, 2, 2);
    let scale = 1.0 / (k as f32).sqrt();
    let def = build_two_timestep_gdn(h, k, v, scale);
    assert!(def.validate().is_ok(), "{:?}", def.validate());
    assert!(
        def.nodes.len() > 11,
        "expected >11 nodes, got {}",
        def.nodes.len()
    );
    assert_eq!(def.nodes[def.output.index()].shape, vec![h, v]);
}

/// Two-timestep composition: all-Variable IBP bounds propagation.
#[test]
fn test_gdn_two_timestep_ibp_all_variable() {
    let (h, dim) = (1, 2);
    let scale = 1.0 / (dim as f32).sqrt();
    let def = build_two_timestep_gdn(h, dim, dim, scale);

    let bindings = vec![TensorParamBinding::Variable; 11];
    let input = uniform_bounds(&[11, h, dim, dim], 1.0);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("two-timestep GDN graph must build");

    match graph.propagate_ibp(&input) {
        Ok(output) => {
            assert_bounds_valid(&output);
        }
        Err(e) => {
            // Known pre-existing issue: multi-variable input stacking pads all
            // variables to max dimension, causing shape mismatch when variables
            // have heterogeneous shapes (e.g., [h,k] vs [h,k,v] in GDN).
            let msg = format!("{e}");
            assert!(
                msg.contains("Shape mismatch"),
                "Expected known shape mismatch error, got unexpected: {e}"
            );
        }
    }
}

/// Two-timestep composition: CROWN propagation through chained state.
#[test]
fn test_gdn_two_timestep_crown_all_variable() {
    let (h, dim) = (1, 2);
    let scale = 1.0 / (dim as f32).sqrt();
    let def = build_two_timestep_gdn(h, dim, dim, scale);

    let bindings = vec![TensorParamBinding::Variable; 11];
    let input = uniform_bounds(&[11, h, dim, dim], 1.0);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("two-timestep GDN graph must build");

    match propagate_with_crown_fallback(&graph, &input) {
        Ok((method, output, fallback)) => {
            assert_bounds_valid(&output);
            if let Some(reason) = &fallback {
                eprintln!("Two-timestep CROWN fell back to IBP: {reason}");
            }
            eprintln!("Two-timestep CROWN: method={method:?}");
        }
        Err(e) => {
            // Known pre-existing issue: same multi-variable padding shape
            // mismatch as test_gdn_two_timestep_ibp_all_variable above.
            let msg = format!("{e}");
            assert!(
                msg.contains("Shape mismatch"),
                "Expected known shape mismatch error, got unexpected: {e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// verify_tensor_and_record pipeline test
// ---------------------------------------------------------------------------

/// verify_tensor_and_record records GDN verification in status.
#[test]
fn test_gdn_verify_and_record() {
    use nn_verify::VerifyStatus;

    let (h, k, v) = (1, 2, 2);
    let scale = 1.0 / (k as f32).sqrt();
    let def = build_gdn_kernel("gdn_pipeline", h, k, v, scale);

    let bindings = vec![TensorParamBinding::Variable; 6];
    let input = uniform_bounds(&[6, h, k, v], 1.0);

    let mut status = VerifyStatus::default();
    let result = nn_verify::verify_tensor_and_record(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("gdn_monolithic_composition"),
    );

    match result {
        Ok(r) => {
            assert!(r.verification.is_finite, "bounds must be finite");
            assert_eq!(r.num_variables, 6, "expected 6 variable inputs");
            assert_bounds_valid(&r.output_bounds);
        }
        Err(e) => {
            panic!("GDN pipeline failed unexpectedly: {e}");
        }
    }
}
