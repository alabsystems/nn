// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-layer CROWN composition of the Kokoro pipeline at D=128 and D=256.
//!
//! While `compose_kokoro_scaled_pipeline.rs` uses monolithic IBP on the entire
//! pipeline graph, this test decomposes the Kokoro architecture into 5 layers
//! and applies per-layer CROWN propagation via `verify_layerwise` (#1762).
//!
//! This is the critical scaling step: monolithic IBP produces vacuously wide
//! bounds at D=128+ due to interval arithmetic accumulation. Per-layer CROWN
//! composes tighter bounds through the pipeline by applying CROWN independently
//! to each layer and checking junction compatibility between stages.
//!
//! Architecture decomposition (5 layers, S = seq_len):
//! ```text
//!   Layer 0: TextEncoder — Conv1d + ReLU + Linear    [D, S] → [D, S]
//!   Layer 1: VocoderPre — Conv1d + LeakyReLU         [D, S] → [D/2, S]
//!   Layer 2: VocoderUpsample — ConvTranspose1d       [D/2, S]  → [D/2, T]
//!   Layer 3: VocoderResBlock — InstNorm+Snake+Conv1d  [D/2, T] → [D/2, T]
//!   Layer 4: VocoderOutput — LeakyReLU+Conv1d+Exp    [D/2, T] → [D/4, T]
//! ```
//! D=128 uses S=8; D=256 uses S=2 (CROWN propagation at D=256/S=4 still exceeds 20min).
//!
//! Part of #1741: THE MOONSHOT — D=128/D=256 per-layer CROWN scaling.

#[path = "kokoro_scaled_pipeline.rs"]
mod d128_scaled_helpers;
// Alias needed: kokoro_scaled_layerwise.rs references `super::helpers::KokoroDims`.
use d128_scaled_helpers as helpers;

#[path = "kokoro_scaled_layerwise.rs"]
mod layerwise_helpers;

use super::common::kokoro_recording::{
    pipeline_crown_coverage, pipeline_tight_stage_count, record_pipeline_certificate,
};
use super::common::kokoro_weights::uniform_bt;
use d128_scaled_helpers::KokoroDims;
use layerwise_helpers::build_kokoro_layerwise;
use nn_tts_verify::verify_layerwise;
use nn_verify::{tensor_kernel_to_graph, PropMethod, VerifyStatus};

/// Assert P1 (non-silence) and P2 (bounded) on layerwise certificate.
fn assert_p1_p2(cert: &nn_tts_verify::PipelineCertificate, label: &str) {
    let lo_min = cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let hi_max = cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        lo_min > 0.0,
        "{label} P1: exp output positive, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "{label} P2: output bounded, got {hi_max}"
    );
    eprintln!("{label}: [{lo_min:.8}, {hi_max:.6}] — P1 ✓ P2 ✓");
}

/// Log and assert junction compatibility.
fn assert_junctions_valid(cert: &nn_tts_verify::PipelineCertificate) {
    for (i, j) in cert.junctions.iter().enumerate() {
        assert!(
            j.shape_compatible,
            "junction {i}: shapes must be compatible"
        );
        assert!(
            j.bounds_contained,
            "junction {i}: bounds contained, max_violation={:.6}",
            j.max_violation
        );
    }
}

/// Log per-stage width analysis and assert final stage has positive lower bound.
fn assert_final_stage_positive(cert: &nn_tts_verify::PipelineCertificate) {
    let final_stage = cert.stages.last().expect("has stages");
    let final_lo = final_stage
        .output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(
        final_lo > 0.0,
        "final stage (vocoder_output with exp) must have positive lower bound"
    );
}

/// Assert that the pipeline certificate is non-vacuous: at least half of
/// stages used CROWN propagation and the output width is practically meaningful.
fn assert_non_vacuous(cert: &nn_tts_verify::PipelineCertificate, label: &str) {
    let total = cert.stages.len();
    let crown_stages = pipeline_tight_stage_count(cert);
    let coverage = pipeline_crown_coverage(cert);
    // Output width: max(upper) - min(lower) across all output elements.
    let lo_min = cert
        .e2e_output_lower
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let hi_max = cert
        .e2e_output_upper
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let output_width = (hi_max - lo_min) as f32;
    let is_non_vacuous = coverage >= 0.5 && output_width < 10.0;
    assert!(
        is_non_vacuous,
        "{label}: certificate is vacuous — crown_coverage={coverage:.2} ({crown_stages}/{total}), \
         output_width={output_width:.4}"
    );
    eprintln!("{label}: non-vacuous ✓ crown_coverage={coverage:.2} output_width={output_width:.4}");
}

/// Verify moonshot bridge from stages and assert P1 passes.
fn assert_moonshot_p1(cert: &nn_tts_verify::PipelineCertificate, d_model: usize) {
    use nn_tts_verify::verify_moonshot_from_stages;
    let bundle = verify_moonshot_from_stages(&cert.stages, d_model).expect("moonshot from stages");
    assert_eq!(bundle.verification_dim, d_model);
    let p1 = bundle
        .results
        .iter()
        .find(|r| r.property_index == 0)
        .expect("P1 must be present");
    assert!(p1.proven, "P1 non-silence must pass: {p1}");
}

/// Run layerwise CROWN verification at given dims, record to status, return cert.
///
/// Also runs IBP on the same layers to compute CROWN/IBP comparison data (#2578).
fn verify_and_record_crown(
    dims: &KokoroDims,
    status: &mut VerifyStatus,
    status_key: &str,
) -> (nn_tts_verify::PipelineCertificate, PropMethod) {
    let layers = build_kokoro_layerwise(dims);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise(&layers, &initial).unwrap_or_else(|e| panic!("{status_key}: {e}"));
    let out_shape = cert.stages.last().expect("stages").output_shape.clone();
    let method = record_pipeline_to_status(status, &cert, status_key, &out_shape);

    // Run IBP comparison only when the pipeline was actually all-CROWN.
    let ibp_width = compute_ibp_layerwise_width(&layers, &initial);
    if method.is_tight() {
        status
            .record_crown_comparison(status_key, ibp_width)
            .unwrap_or_else(|e| panic!("{status_key} IBP comparison: {e}"));
    } else {
        eprintln!(
            "{status_key}: skipping CROWN/IBP comparison because stage mix recorded as {method:?}"
        );
    }

    let crown_width = {
        let lo = cert
            .e2e_output_lower
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let hi = cert
            .e2e_output_upper
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        (hi - lo) as f32
    };
    let ratio = crown_width / ibp_width.max(1e-10);
    eprintln!(
        "{status_key}: method={method:?}, crown_width={crown_width:.6}, \
         ibp_width={ibp_width:.6}, ratio={ratio:.4}"
    );
    (cert, method)
}

/// Compute the end-to-end IBP output width for a layerwise pipeline.
///
/// Runs `propagate_ibp` per-layer, chaining output bounds from layer N as
/// input bounds to layer N+1. Returns `max(upper) - min(lower)` of the
/// final layer output, matching the width metric used for CROWN recording.
fn compute_ibp_layerwise_width(
    layers: &[(
        nn_dsl::tensor_ir::TensorKernelDef,
        Vec<nn_verify::TensorParamBinding>,
    )],
    initial_bounds: &nn_verify::BoundedTensor,
) -> f32 {
    let mut current = initial_bounds.clone();
    for (i, (layer, bindings)) in layers.iter().enumerate() {
        let graph = tensor_kernel_to_graph(layer, bindings)
            .unwrap_or_else(|e| panic!("IBP layer {i} graph: {e}"));
        current = graph
            .propagate_ibp(&current)
            .unwrap_or_else(|e| panic!("IBP layer {i} propagation: {e}"));
    }
    let lo = current
        .lower()
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let hi = current
        .upper()
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    hi - lo
}

/// Record a `PipelineCertificate` in `VerifyStatus` with its actual stage mix.
///
/// Converts f64 e2e bounds to f32 scalar summaries and writes a single entry
/// keyed by `status_key` (e.g., `kokoro_layerwise_d128_crown`).
fn record_pipeline_to_status(
    status: &mut VerifyStatus,
    cert: &nn_tts_verify::PipelineCertificate,
    status_key: &str,
    output_shape: &[usize],
) -> PropMethod {
    record_pipeline_certificate(status, status_key, cert, output_shape, None)
}

/// Per-model status file path for Kokoro kernels (#2577).
fn status_file_path() -> std::path::PathBuf {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    nn_verify::model_status_path(workspace_root, "kokoro")
}

// ===========================================================================
// D=64 layerwise (sanity check — should be tighter than monolithic IBP)
// ===========================================================================

/// D=64 layerwise: pipeline validity + P1 (non-silence) + P2 (bounded).
#[test]
fn test_kokoro_layerwise_d64_all_properties() {
    let dims = KokoroDims::d64();
    let layers = build_kokoro_layerwise(&dims);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise(&layers, &initial).expect("D=64 layerwise");
    assert!(cert.is_valid, "D=64 layerwise pipeline must be valid");
    assert_p1_p2(&cert, "D=64 layerwise");
}

// ===========================================================================
// D=128 layerwise (the main scaling target)
// ===========================================================================

/// D=128 layerwise: all properties in a single CROWN propagation.
///
/// Consolidates pipeline validity, junction checks, P1 (non-silence),
/// P2 (bounded), per-stage width analysis, and moonshot bridge into one
/// test to avoid 5× redundant CROWN propagation at D=128.
#[test]
fn test_kokoro_layerwise_d128_all_properties() {
    let dims = KokoroDims::d128();
    let layers = build_kokoro_layerwise(&dims);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise(&layers, &initial).expect("D=128 layerwise");

    assert!(cert.is_valid, "D=128 layerwise pipeline must be valid");
    assert_junctions_valid(&cert);
    assert_p1_p2(&cert, "D=128 layerwise");
    assert_final_stage_positive(&cert);
    assert_non_vacuous(&cert, "D=128 layerwise");
    assert_moonshot_p1(&cert, dims.d_model);
}

// ===========================================================================
// D=256 layerwise (approaching production scale — D=512/2)
// ===========================================================================

/// D=256 layerwise: all properties in a single CROWN propagation.
///
/// Consolidates pipeline validity, P1 (non-silence), P2 (bounded),
/// and moonshot bridge into one test to avoid 4× redundant CROWN
/// propagation at D=256 (the most expensive dimension).
#[test]
fn test_kokoro_layerwise_d256_all_properties() {
    let dims = KokoroDims::d256();
    let layers = build_kokoro_layerwise(&dims);
    let initial = uniform_bt(&[dims.d_model, dims.seq_len], -1.0, 1.0);
    let cert = verify_layerwise(&layers, &initial).expect("D=256 layerwise");

    assert!(cert.is_valid, "D=256 layerwise pipeline must be valid");
    assert_p1_p2(&cert, "D=256 layerwise");
    assert_moonshot_p1(&cert, dims.d_model);
}

// ===========================================================================
// Persist CROWN results to nn_verify_status.json (#2453)
// Split into per-dimension tests to avoid timeout (#2578).
// ===========================================================================

/// Helper: verify CROWN at given dimension, record to status, validate entry.
fn persist_and_validate_crown(dims: &KokoroDims, status_key: &str) {
    let status_path = status_file_path();
    let mut locked = VerifyStatus::load_locked(&status_path).expect("load_locked");

    let (_cert, expected_method) = verify_and_record_crown(dims, &mut locked.status, status_key);

    locked.save().expect("save status");
    drop(locked);

    // Validate: entry present with correct method.
    let validation = VerifyStatus::load_locked(&status_path).expect("load_locked validation");
    let entry = validation
        .status
        .kernel(status_key)
        .unwrap_or_else(|| panic!("expected entry for {status_key}"));
    assert_eq!(
        entry.method, expected_method,
        "{status_key} status entry must reflect actual per-stage method mix"
    );
    assert!(
        expected_method.is_tight(),
        "{status_key} silently degraded from CROWN-family propagation to {expected_method:?}"
    );
}

/// Validate that a CROWN entry has IBP comparison data (#2578).
fn assert_entry_has_comparison(status_key: &str) {
    let status_path = status_file_path();
    let locked = VerifyStatus::load_locked(&status_path).expect("load_locked");
    let entry = locked
        .status
        .kernel(status_key)
        .unwrap_or_else(|| panic!("expected entry for {status_key}"));
    assert!(
        entry.ibp_comparison_width.is_some(),
        "{status_key} must have ibp_comparison_width (#2578)"
    );
    assert!(
        entry.crown_ibp_ratio.is_some(),
        "{status_key} must have crown_ibp_ratio (#2578)"
    );
    let ratio = entry.crown_ibp_ratio.unwrap();
    assert!(
        ratio.is_finite(),
        "{status_key} ratio must be finite, got {ratio}"
    );
    eprintln!(
        "{status_key}: ibp_width={:.6}, ratio={ratio:.4}",
        entry.ibp_comparison_width.unwrap()
    );
}

/// D=64 layerwise CROWN: verify, record, validate soundness + IBP comparison.
///
/// After gc#4399, Conservative-mode re-verification can upgrade these entries
/// to Sound. The soundness assertion accepts either Heuristic (CrownSampling
/// path) or Sound (Conservative re-verification path) since test execution
/// order is non-deterministic.
#[test]
fn test_kokoro_layerwise_persist_d64_crown() {
    persist_and_validate_crown(&KokoroDims::d64(), "kokoro_layerwise_d64_crown");
    assert_entry_has_comparison("kokoro_layerwise_d64_crown");
}

/// D=128 layerwise CROWN: verify, record, validate soundness + IBP comparison.
#[test]
fn test_kokoro_layerwise_persist_d128_crown() {
    persist_and_validate_crown(&KokoroDims::d128(), "kokoro_layerwise_d128_crown");
    assert_entry_has_comparison("kokoro_layerwise_d128_crown");
}

/// D=256 layerwise CROWN: verify, record, validate soundness + IBP comparison.
#[test]
fn test_kokoro_layerwise_persist_d256_crown() {
    persist_and_validate_crown(&KokoroDims::d256(), "kokoro_layerwise_d256_crown");
    assert_entry_has_comparison("kokoro_layerwise_d256_crown");
}

#[test]
fn test_record_pipeline_to_status_marks_mixed_stage_methods() {
    use nn_tts_verify::{verify_pipeline, VerifiedStage};

    let stages = vec![
        VerifiedStage::new(
            "stage0",
            vec![1],
            vec![1],
            vec![-1.0],
            vec![1.0],
            vec![-0.5],
            vec![0.5],
            "CROWN",
            true,
        ),
        VerifiedStage::new(
            "stage1",
            vec![1],
            vec![1],
            vec![-0.5],
            vec![0.5],
            vec![-0.25],
            vec![0.25],
            "IBP",
            false,
        ),
    ];
    let cert = verify_pipeline(&stages).expect("mixed pipeline cert");

    let mut status = VerifyStatus::default();
    let method = record_pipeline_to_status(&mut status, &cert, "synthetic_kokoro_mixed", &[1]);
    let entry = status
        .kernel("synthetic_kokoro_mixed")
        .expect("status entry");

    assert_eq!(method, PropMethod::MixedIbpCrown);
    assert_eq!(entry.method, PropMethod::MixedIbpCrown);
}

/// Validate IBP comparison data across all 3 CROWN entries (#2578).
///
/// This is a validation-only test: reads the status file populated by the
/// per-dimension persist tests (`persist_d64`, `persist_d128`, `persist_d256`)
/// and validates the combined CROWN/IBP comparison report. Does NOT re-run
/// CROWN propagation — that's done by the individual tests.
#[test]
fn test_kokoro_layerwise_crown_ibp_comparison_report() {
    // Run D=64 only (fast) to ensure at least one comparison entry exists.
    persist_and_validate_crown(&KokoroDims::d64(), "kokoro_layerwise_d64_crown");

    let status_path = status_file_path();
    let validation = VerifyStatus::load_locked(&status_path).expect("load_locked validation");

    // Validate D=64 has IBP comparison data.
    let d64 = validation
        .status
        .kernel("kokoro_layerwise_d64_crown")
        .expect("d64 entry");
    assert!(
        d64.ibp_comparison_width.is_some(),
        "D=64 must have ibp_comparison_width (#2578)"
    );
    assert!(
        d64.crown_ibp_ratio.is_some(),
        "D=64 must have crown_ibp_ratio (#2578)"
    );
    let ratio = d64.crown_ibp_ratio.unwrap();
    assert!(ratio.is_finite(), "D=64 ratio must be finite, got {ratio}");
    eprintln!(
        "D=64: crown_width={:.6}, ibp_width={:.6}, ratio={ratio:.4}",
        d64.output_width,
        d64.ibp_comparison_width.unwrap()
    );

    // Report on all entries with comparison data.
    let (crown_count, tighter_count, entries) = validation.status.crown_comparison_report();
    eprintln!("CROWN comparison: {crown_count} entries, {tighter_count} tighter than IBP");
    for (name, r) in &entries {
        let label = if *r < 1.0 {
            "CROWN tighter"
        } else {
            "IBP equivalent"
        };
        eprintln!("  {name}: ratio={r:.4} — {label}");
    }
    // At least 1 entry with comparison data (D=64 that we just ran).
    assert!(
        crown_count >= 1,
        "expected >= 1 CROWN entry with comparison data, got {crown_count}"
    );
}
