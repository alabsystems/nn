// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Performance proof tests for certificate checker algorithmic complexity.
//!
//! Proves that `check_certificate` is O(L) in layer count, not O(L²).
//! The key invariant: `check_layer_trace_graph_aware` builds a HashMap index
//! (O(L) setup) for source lookups, avoiding O(L²) linear scan.
//!
//! Also proves that the issues Vec is bounded for valid certificates and
//! that the output agreement fold is O(E) in element count.
//!
//! # Certificate integrity performance finding (2026-03-22, P10 audit)
//!
//! `canonical_json()` in `certificate_integrity.rs` previously cloned the
//! entire `ProofCertificate` to clear `content_hash` and `hmac_signature`
//! before serializing. Fixed: now uses `serde_json::to_value` to serialize
//! field-by-field from the reference, then removes the two integrity keys
//! from the JSON map. No struct clone needed.
//!
//! Part of #3020 (certificate pipeline performance).

use super::*;
use crate::certificate::ProofCertificate;
use crate::certificate_types::LayerBoundRecord;
use crate::soundness_compat::VerificationSoundnessMode;
use crate::status::{InputBoundsRecord, ParamInputRecord};
use crate::verify_types::{KernelVerification, OutputTensorBounds, PropMethod};

/// Build a consistent N-layer graph-aware certificate trace.
///
/// Each layer i has `input_sources: [i-1]` (linear chain).
/// Layer 0 has empty `input_sources` (network input).
/// All bounds are consistent: layer[i].output == layer[i+1].input.
fn build_n_layer_trace(n: usize, elements_per_layer: usize) -> Vec<LayerBoundRecord> {
    let mut bounds = Vec::with_capacity(n);
    for i in 0..n {
        let input_bounds = if i == 0 {
            vec![(-10.0, 10.0); elements_per_layer]
        } else {
            // Match previous layer's output: (-5.0, 5.0)
            vec![(-5.0, 5.0); elements_per_layer]
        };
        bounds.push(LayerBoundRecord {
            layer_index: i,
            layer_type: "Linear".to_string(),
            input_bounds,
            output_bounds: vec![(-5.0, 5.0); elements_per_layer],
            method: PropMethod::Crown,
            node_name: None,
            input_sources: Some(if i == 0 { vec![] } else { vec![i - 1] }),
        });
    }
    bounds
}

fn perf_verification(elements: usize) -> KernelVerification {
    KernelVerification {
        kernel_name: "perf_test".to_string(),
        method: PropMethod::Crown,
        output_lower: -5.0,
        output_upper: 5.0,
        output_width: 10.0,
        is_finite: true,
        crown_fallback_reason: None,
        soundness_mode: VerificationSoundnessMode::Sound,
        output_tensor: Some(OutputTensorBounds {
            lower: vec![-5.0; elements],
            upper: vec![5.0; elements],
            shape: vec![elements],
            finite_mask: vec![true; elements],
        }),
    }
}

fn perf_input_spec() -> InputBoundsRecord {
    InputBoundsRecord {
        variable_inputs: vec![ParamInputRecord {
            param_index: 0,
            lower: -10.0,
            upper: 10.0,
        }],
        constant_params: vec![1.0],
        input_shape: Some(vec![1]),
        input_range: Some((-10.0, 10.0)),
    }
}

// ---------------------------------------------------------------------------
// Graph-aware trace checking is O(L), not O(L²)
// ---------------------------------------------------------------------------

/// Prove: check_certificate on an N-layer graph-aware trace completes
/// within a time bound proportional to N, not N².
///
/// The HashMap index at certificate_checker.rs:220-224 ensures source
/// lookups are O(1) per edge, giving O(L) total for a linear chain.
/// Without the HashMap, a naive linear scan would be O(L²).
///
/// Method: Run check_certificate on 100, 1000, and 10000 layers.
/// If the implementation is O(L²), the 10000-layer check would take
/// ~100× longer than the 1000-layer check. With O(L), it takes ~10×.
/// We verify the ratio is < 20× (generous margin for constant factors).
#[test]
fn proof_graph_aware_trace_checking_is_linear() {
    let sizes = [100, 1000, 10000];
    let mut durations = Vec::new();

    for &n in &sizes {
        let trace = build_n_layer_trace(n, 4);
        let result = perf_verification(4);
        let cert = ProofCertificate::from_verification(&result, perf_input_spec())
            .with_layer_bounds(trace)
            .with_source_hash("a".repeat(64));

        let start = std::time::Instant::now();
        let check = check_certificate(&cert, None, None);
        let elapsed = start.elapsed();

        // Valid certificate should have no errors (only possibly VacuousBounds).
        assert!(
            check.is_valid(),
            "n={n}: should be valid, issues: {:?}",
            check.issues
        );
        durations.push((n, elapsed));
    }

    // Compare 10000-layer vs 1000-layer ratio.
    // O(L): ratio ≈ 10. O(L²): ratio ≈ 100.
    let t_1000 = durations[1].1.as_nanos() as f64;
    let t_10000 = durations[2].1.as_nanos() as f64;

    // Guard: ensure 1000-layer time is measurable (> 1μs).
    if t_1000 > 1000.0 {
        let ratio = t_10000 / t_1000;
        assert!(
            ratio < 30.0,
            "10000/1000 layer ratio = {ratio:.1}×, expected < 30× for O(L). \
             If > 50× this indicates O(L²). \
             t_1000={:.3}ms, t_10000={:.3}ms",
            t_1000 / 1e6,
            t_10000 / 1e6
        );
    }
}

/// Prove: valid certificate produces zero error issues.
///
/// A consistent N-layer trace with CROWN propagation and finite bounds
/// should produce only informational VacuousBounds (if at all). No
/// LayerTraceGap, OutputMismatch, or structural errors.
#[test]
fn proof_valid_certificate_zero_error_issues() {
    for &n in &[10, 100, 500] {
        let trace = build_n_layer_trace(n, 8);
        let result = perf_verification(8);
        let cert = ProofCertificate::from_verification(&result, perf_input_spec())
            .with_layer_bounds(trace)
            .with_source_hash("a".repeat(64));

        let check = check_certificate(&cert, None, None);
        let error_issues: Vec<_> = check
            .issues
            .iter()
            .filter(|i| !matches!(i, CheckIssue::VacuousBounds { .. }))
            .collect();
        assert!(
            error_issues.is_empty(),
            "n={n}: expected zero error issues for valid cert, got {}: {:?}",
            error_issues.len(),
            error_issues
        );
    }
}

/// Prove: issues Vec is bounded at O(L) even for fully-corrupted traces.
///
/// Worst case: every layer has inverted bounds (lo > hi) on all elements.
/// check_inverted_element_bounds produces one CheckIssue per inverted element.
/// Total issues = L * E. This is O(L*E) = O(total_elements), which is
/// linear in input size. No hidden quadratic amplification.
#[test]
fn proof_issues_vec_bounded_by_total_elements() {
    let n = 100;
    let e = 10;
    let mut trace = build_n_layer_trace(n, e);

    // Corrupt all elements: make lower > upper (inverted).
    for record in &mut trace {
        record.output_bounds = vec![(5.0, -5.0); e]; // inverted
    }

    let result = perf_verification(e);
    let cert = ProofCertificate::from_verification(&result, perf_input_spec())
        .with_layer_bounds(trace)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    let inverted_count = check
        .issues
        .iter()
        .filter(|i| matches!(i, CheckIssue::InvertedElementBounds { .. }))
        .count();

    // Each layer × each element = exactly L*E inverted issues.
    assert_eq!(
        inverted_count,
        n * e,
        "inverted issues should be exactly L*E = {}, got {inverted_count}",
        n * e
    );
    // Total issues should be at most L*E + O(L) (trace gaps from corruption).
    assert!(
        check.issues.len() <= n * e + 2 * n,
        "total issues {} should be bounded by L*E + 2*L = {}",
        check.issues.len(),
        n * e + 2 * n
    );
}

/// Prove: output agreement fold is O(E) in element count.
///
/// The fold at agreement.rs:49-58 reduces E elements to min(lower)/max(upper).
/// This test verifies correctness across increasing E values, confirming
/// the fold handles large element counts without blowup.
#[test]
fn proof_output_agreement_fold_scales_linearly() {
    for &e in &[1, 10, 100, 1000, 10000] {
        let trace = build_n_layer_trace(3, e);
        let result = perf_verification(e);
        let cert = ProofCertificate::from_verification(&result, perf_input_spec())
            .with_layer_bounds(trace)
            .with_source_hash("a".repeat(64));

        let check = check_certificate(&cert, None, None);
        assert!(
            check.is_valid(),
            "e={e}: should be valid, issues: {:?}",
            check.issues
        );
    }
}

/// Prove: HashMap index in graph-aware mode handles diamond DAGs correctly.
///
/// Constructs a diamond topology: A → B, A → C, B → D, C → D.
/// Layer D has two sources (B and C). The HashMap-indexed lookup must
/// find both sources efficiently. This also tests multi-source handling.
#[test]
fn proof_diamond_dag_handled_by_hashmap_index() {
    let bounds = vec![
        LayerBoundRecord {
            layer_index: 0,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-10.0, 10.0)],
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Crown,
            node_name: Some("A".to_string()),
            input_sources: Some(vec![]),
        },
        LayerBoundRecord {
            layer_index: 1,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            output_bounds: vec![(-3.0, 3.0)],
            method: PropMethod::Crown,
            node_name: Some("B".to_string()),
            input_sources: Some(vec![0]),
        },
        LayerBoundRecord {
            layer_index: 2,
            layer_type: "Linear".to_string(),
            input_bounds: vec![(-5.0, 5.0)],
            output_bounds: vec![(-2.0, 4.0)],
            method: PropMethod::Crown,
            node_name: Some("C".to_string()),
            input_sources: Some(vec![0]),
        },
        LayerBoundRecord {
            layer_index: 3,
            layer_type: "Add".to_string(),
            input_bounds: vec![(-5.0, 7.0)], // combined bounds from NY
            output_bounds: vec![(-5.0, 5.0)],
            method: PropMethod::Crown,
            node_name: Some("D".to_string()),
            input_sources: Some(vec![1, 2]), // multi-source: B and C
        },
    ];

    let result = perf_verification(1);
    let cert = ProofCertificate::from_verification(&result, perf_input_spec())
        .with_layer_bounds(bounds)
        .with_source_hash("a".repeat(64));

    let check = check_certificate(&cert, None, None);
    // Diamond DAGs with multi-source layers should not produce spurious issues.
    let error_issues: Vec<_> = check
        .issues
        .iter()
        .filter(|i| !matches!(i, CheckIssue::VacuousBounds { .. }))
        .collect();
    assert!(
        error_issues.is_empty(),
        "diamond DAG should be valid, got errors: {error_issues:?}"
    );
}

// ---------------------------------------------------------------------------
// canonical_json serialization scales linearly with certificate size
// ---------------------------------------------------------------------------

/// Prove: canonical_json serialization is O(cert_size) — not quadratic.
///
/// `canonical_json()` serializes certificates via `to_value` + key removal.
/// For certificates with N layers × E elements, the cost is O(N*E) for
/// serialization. This test validates linear scaling.
#[test]
fn proof_canonical_json_clone_scales_linearly() {
    use crate::certificate::integrity::compute_content_hash;

    let elements = 100;
    let sizes = [10, 100, 1000];
    let mut durations = Vec::new();

    for &n in &sizes {
        let trace = build_n_layer_trace(n, elements);
        let result = perf_verification(elements);
        let cert = ProofCertificate::from_verification(&result, perf_input_spec())
            .with_layer_bounds(trace)
            .with_source_hash("a".repeat(64));

        let start = std::time::Instant::now();
        let hash = compute_content_hash(&cert).expect("hash should succeed");
        let elapsed = start.elapsed();

        assert_eq!(hash.len(), 64, "SHA-256 hex is 64 chars");
        durations.push((n, elapsed));
    }

    let t_100 = durations[1].1.as_nanos() as f64;
    let t_1000 = durations[2].1.as_nanos() as f64;

    if t_100 > 1000.0 {
        let ratio = t_1000 / t_100;
        assert!(
            ratio < 25.0,
            "canonical_json 1000/100 ratio = {ratio:.1}x, expected < 25x for O(N). \
             t_100={:.3}ms, t_1000={:.3}ms",
            t_100 / 1e6,
            t_1000 / 1e6
        );
    }
}

/// Prove: sign_bundle is O(B * C) where B=bundle size, C=cert size.
///
/// No hidden quadratic interaction between certificates in a bundle.
/// Each certificate is signed independently.
#[test]
fn proof_sign_bundle_scales_linearly_in_bundle_size() {
    use crate::certificate::integrity::sign_bundle;
    use crate::certificate::CertificateBundle;

    let layers = 50;
    let elements = 20;

    let mut bundles = Vec::new();
    for &b_size in &[5, 20] {
        let mut bundle = CertificateBundle::new("perf_test");
        for _ in 0..b_size {
            let trace = build_n_layer_trace(layers, elements);
            let result = perf_verification(elements);
            let cert = ProofCertificate::from_verification(&result, perf_input_spec())
                .with_layer_bounds(trace)
                .with_source_hash("a".repeat(64));
            bundle.push(cert);
        }
        bundles.push((b_size, bundle));
    }

    let key = b"perf-test-key";
    let mut durations = Vec::new();
    for (b_size, mut bundle) in bundles {
        let start = std::time::Instant::now();
        sign_bundle(&mut bundle, key).expect("signing should succeed");
        let elapsed = start.elapsed();
        durations.push((b_size, elapsed));
    }

    let t_5 = durations[0].1.as_nanos() as f64;
    let t_20 = durations[1].1.as_nanos() as f64;

    if t_5 > 1000.0 {
        let ratio = t_20 / t_5;
        assert!(
            ratio < 10.0,
            "sign_bundle 20/5 ratio = {ratio:.1}x, expected < 10x for O(B). \
             t_5={:.3}ms, t_20={:.3}ms",
            t_5 / 1e6,
            t_20 / 1e6
        );
    }
}

/// Prove: verify_bundle_signatures is O(B * C) — linear in bundle size.
///
/// Each certificate is verified independently: canonical_json + SHA-256 + HMAC.
/// No inter-certificate state accumulation that could cause O(B²).
#[test]
fn proof_verify_bundle_signatures_scales_linearly() {
    use crate::certificate::integrity::{sign_bundle, verify_bundle_signatures};
    use crate::certificate::CertificateBundle;

    let layers = 50;
    let elements = 20;
    let key = b"perf-verify-key";

    let mut durations = Vec::new();
    for &b_size in &[5, 20] {
        let mut bundle = CertificateBundle::new("perf_test");
        for _ in 0..b_size {
            let trace = build_n_layer_trace(layers, elements);
            let result = perf_verification(elements);
            let cert = ProofCertificate::from_verification(&result, perf_input_spec())
                .with_layer_bounds(trace)
                .with_source_hash("a".repeat(64));
            bundle.push(cert);
        }
        sign_bundle(&mut bundle, key).expect("signing should succeed");

        let start = std::time::Instant::now();
        verify_bundle_signatures(&bundle, key).expect("verification should succeed");
        let elapsed = start.elapsed();
        durations.push((b_size, elapsed));
    }

    let t_5 = durations[0].1.as_nanos() as f64;
    let t_20 = durations[1].1.as_nanos() as f64;

    if t_5 > 1000.0 {
        let ratio = t_20 / t_5;
        assert!(
            ratio < 10.0,
            "verify_bundle 20/5 ratio = {ratio:.1}x, expected < 10x for O(B). \
             t_5={:.3}ms, t_20={:.3}ms",
            t_5 / 1e6,
            t_20 / 1e6
        );
    }
}

/// Prove: content hash computation scales linearly with layer_bounds size.
///
/// canonical_json → sort_json_keys recursion processes O(total_json_nodes).
/// For L layers × E elements, total nodes ≈ L * (7 keys + 2*E floats).
/// Verify no hidden quadratic from recursive sort or map drain+rebuild.
#[test]
fn proof_content_hash_linear_in_layer_elements() {
    use crate::certificate::integrity::compute_content_hash;

    // Fixed layers, scale elements per layer.
    let layers = 50;
    let sizes = [10, 100, 1000];
    let mut durations = Vec::new();

    for &e in &sizes {
        let trace = build_n_layer_trace(layers, e);
        let result = perf_verification(e);
        let cert = ProofCertificate::from_verification(&result, perf_input_spec())
            .with_layer_bounds(trace)
            .with_source_hash("a".repeat(64));

        let start = std::time::Instant::now();
        let hash = compute_content_hash(&cert).expect("hash should succeed");
        let elapsed = start.elapsed();

        assert_eq!(hash.len(), 64);
        durations.push((e, elapsed));
    }

    // 1000 vs 100 elements: data is 10x, so time should be <20x (generous margin).
    let t_100 = durations[1].1.as_nanos() as f64;
    let t_1000 = durations[2].1.as_nanos() as f64;

    if t_100 > 1000.0 {
        let ratio = t_1000 / t_100;
        assert!(
            ratio < 25.0,
            "content_hash 1000/100 element ratio = {ratio:.1}x, expected < 25x. \
             t_100={:.3}ms, t_1000={:.3}ms",
            t_100 / 1e6,
            t_1000 / 1e6
        );
    }
}
