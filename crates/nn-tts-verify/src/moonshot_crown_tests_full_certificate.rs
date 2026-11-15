// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Full 8-property moonshot certificate tests (#1741 P7+P8 gap).
//!
//! These tests exercise the complete moonshot certification pipeline:
//! - P1-P6 via CROWN bundle (reusing composed pipeline from temporal_composed)
//! - P8 via `with_smt_results()` (ay SMT verification evidence)
//!
//! P7 (memory safety) has had no evidence source since the Kani harnesses
//! were retired (2026-09-29), so it stays `None` and no certificate is
//! `all_proven`.

use super::*;
use crate::moonshot::{MoonshotCertificate, SmtVerificationEvidence};
use crate::MoonshotStatus;

/// ay-proven kernel count from AY_PROVEN_KERNELS.
/// Verify: `ay_proven_kernel_names().len()` — currently 20 entries.
const AY_PROVEN_KERNEL_COUNT: usize = 20;

/// Build a `MoonshotCertificate` with every enrichable property enriched.
///
/// P1-P6 come from the CROWN bundle (D=192 composed pipeline).
/// P8 comes from ay SMT evidence (kernel-level proofs).
fn full_certificate(smt_proven: usize, smt_total: usize) -> MoonshotCertificate {
    let status = MoonshotStatus::from_repo();
    let cert = MoonshotCertificate::from_status(
        &status,
        "dvoice-kokoro-v1",
        "English text, ≤50 words",
        "abc123def456",
    );

    // Build CROWN bundle for P1-P6 at D=192.
    let dim = 192;
    let stages = vec![
        VerifiedStage {
            name: "text_encoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.8; dim],
            output_upper: vec![0.8; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
        VerifiedStage {
            name: "prosody_predictor".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.5; dim],
            output_upper: vec![0.5; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
        VerifiedStage {
            name: "vocoder".to_string(),
            input_lower: vec![-1.0; dim],
            input_upper: vec![1.0; dim],
            output_lower: vec![-0.3; dim],
            output_upper: vec![0.3; dim],
            input_shape: vec![1, dim],
            output_shape: vec![1, dim],
            method: "CROWN".to_string(),
            is_sound: true,
        },
    ];

    let pipeline_cert = crate::pipeline::verify_pipeline(&stages).expect("pipeline must compose");

    let timing_cert = TimingCertificate {
        bounds_cert: pipeline_cert.clone(),
        cost_profiles: stages
            .iter()
            .map(|s| crate::cost_model::LayerCostProfile {
                layer_name: s.name.clone(),
                flops: 1_000_000,
                memory_bytes: 4 * dim as u64,
                estimated_time_us: 15_000.0,
                measured_time_us: None,
            })
            .collect(),
        worst_case_time_us: 45_000.0,
        total_flops: 3_000_000,
        total_memory_bytes: 12 * dim as u64,
        hardware_name: "M4 Max".to_string(),
        timing_bound_us: 100_000.0,
        timing_bound_met: true,
        overall_passed: true,
        peak_memory: None,
    };

    let norm_val = 1.0 / (dim as f64).sqrt();
    let speaker_ev = speaker::speaker_evidence(
        dim,
        vec![norm_val - 0.01; dim],
        vec![norm_val + 0.01; dim],
        vec![norm_val; dim],
        0.3,
        true,
    );

    let attn_cert = crate::monotonicity::AttentionMonotonicityCertificate {
        decoder_steps: 50,
        encoder_positions: 50,
        min_margin: 0.5,
        is_proven: true,
        row_margins: vec![0.5; 50],
        input_bound: 1.0,
        propagation_mode: "CROWN".to_string(),
    };

    // Enrich P1-P6 via CROWN bundle.
    let bundle = verify_all_crown_properties_with_attention(
        &pipeline_cert,
        &timing_cert,
        &speaker_ev,
        Some(&attn_cert),
        dim,
    );
    let cert = cert.with_crown_results(&bundle);

    // Enrich P8 with ay SMT evidence (dynamic kernel names from AY_PROVEN_KERNELS).
    let ay_names = ay_proven_kernel_names();
    let smt_evidence = SmtVerificationEvidence {
        kernels_proven: smt_proven,
        kernels_total: smt_total,
        proven_kernel_names: ay_names
            .iter()
            .take(smt_proven)
            .map(ToString::to_string)
            .collect(),
        all_proven: smt_proven == smt_total && smt_total > 0,
    };
    cert.with_smt_results(&smt_evidence)
}

// ---------------------------------------------------------------------------
// Full 8-property certificate tests
// ---------------------------------------------------------------------------

/// Every property but P7 proven at D=192 with real ay counts.
#[test]
fn test_full_certificate_all_but_p7_proven() {
    let cert = full_certificate(AY_PROVEN_KERNEL_COUNT, AY_PROVEN_KERNEL_COUNT);

    assert!(
        !cert.all_proven,
        "P7 has no evidence, so the certificate is not all-proven"
    );

    // P1-P6 should be CrownProven.
    for i in 0..6 {
        assert_eq!(
            cert.properties[i].level,
            VerificationLevel::CrownProven,
            "P{} must be CrownProven",
            i + 1,
        );
    }

    // P7 (memory safety) has no evidence source.
    assert_eq!(cert.properties[6].level, VerificationLevel::None);
    assert!(cert.properties[6].proof_artifacts.is_empty());

    // P8 should be SmtProven.
    assert_eq!(
        cert.properties[7].level,
        VerificationLevel::SmtProven,
        "P8 must be SmtProven"
    );
    assert_eq!(
        cert.properties[7].bound_value,
        Some(AY_PROVEN_KERNEL_COUNT as f64),
    );
}

/// P8 with partial ay coverage — level should be Empirical.
#[test]
fn test_full_certificate_p8_partial_smt() {
    let cert = full_certificate(
        9, // 9 of 14 proven
        AY_PROVEN_KERNEL_COUNT,
    );

    // P8 should be Empirical (not all proven).
    assert_eq!(
        cert.properties[7].level,
        VerificationLevel::Empirical,
        "partial ay → Empirical"
    );
    assert_eq!(cert.properties[7].bound_value, Some(9.0));

    assert!(!cert.all_proven, "partial ay means not all_proven");
}

/// Certificate model metadata is correct.
#[test]
fn test_full_certificate_metadata() {
    let cert = full_certificate(AY_PROVEN_KERNEL_COUNT, AY_PROVEN_KERNEL_COUNT);

    assert_eq!(cert.model_name, "dvoice-kokoro-v1");
    assert_eq!(cert.input_specification, "English text, ≤50 words");
    assert_eq!(cert.source_hash, "abc123def456");
    assert_eq!(cert.properties.len(), 8);
    assert_eq!(cert.verification_dim, Some(192));
}

/// ay SMT evidence assumptions include QF_LRA reference.
#[test]
fn test_full_certificate_p8_assumptions() {
    let cert = full_certificate(AY_PROVEN_KERNEL_COUNT, AY_PROVEN_KERNEL_COUNT);

    let p8 = &cert.properties[7];
    assert!(
        p8.assumptions.iter().any(|a: &String| a.contains("QF_LRA")),
        "P8 assumptions should mention QF_LRA: {:?}",
        p8.assumptions
    );
}

// ---------------------------------------------------------------------------
// Phase 15: Bridge-integrated certificate tests
// ---------------------------------------------------------------------------

/// Find workspace root by walking up from the current file's directory.
fn workspace_crates_dir() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR points to crates/nn-tts-verify/
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // workspace root is two levels up: crates/nn-tts-verify/ -> crates/ -> root
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    workspace_root.join("crates")
}

/// AY_PROVEN_KERNEL_COUNT must match ay_proven_kernel_names().len().
#[test]
fn test_ay_kernel_count_consistent_with_constant() {
    let names = ay_proven_kernel_names();
    assert_eq!(
        names.len(),
        AY_PROVEN_KERNEL_COUNT,
        "AY_PROVEN_KERNEL_COUNT ({}) must match ay_proven_kernel_names().len() ({})",
        AY_PROVEN_KERNEL_COUNT,
        names.len(),
    );
}

// ---------------------------------------------------------------------------
// D=512 precomputed production bounds + audio clamp (Part of #2463)
// ---------------------------------------------------------------------------

/// Full 8-property moonshot at D=512 from precomputed production IBP bounds.
///
/// Uses the same production bounds as the nn-verify compose test, but
/// constructs stages via `VerifiedStage::new` (no BoundedTensor/nn-verify dep).
/// Audio clamp produces deterministic [-1, 1] output, closing P2 and P6.
///
/// Precomputed bounds from `kokoro_production_moonshot_2stage` in
/// nn_verify_status_kokoro.json (production weights, main zone):
///   Input: token IDs [0, 177], shape [1, 4]
///   Output (PP): [-172.49582, 172.67929], shape [2560]
///
/// Part of #2463.
#[test]
fn test_production_d512_8prop_precomputed_audio_clamp() {
    use crate::moonshot::{FullCertificateBuilder, SmtVerificationEvidence};

    let dim = 2560;
    let te_range = 10.0_f64;
    let pp_range = 172.7_f64;

    // Stages 1-2: TextEncoder → ProsodyPredictor (precomputed production IBP).
    let stage1 = VerifiedStage::new(
        "text_encoder",
        vec![1, 4],
        vec![dim],
        vec![0.0; 4],
        vec![177.0; 4],
        vec![-te_range; dim],
        vec![te_range; dim],
        "IBP",
        false,
    );
    let stage2 = VerifiedStage::new(
        "prosody_predictor",
        vec![dim],
        vec![dim],
        vec![-te_range; dim],
        vec![te_range; dim],
        vec![-pp_range; dim],
        vec![pp_range; dim],
        "IBP",
        false,
    );

    // Stage 3: Audio clamp — deterministic [-1, 1] output.
    let stage_clip = VerifiedStage::new(
        "audio_clamp",
        vec![dim],
        vec![dim],
        vec![-pp_range; dim],
        vec![pp_range; dim],
        vec![-1.0; dim],
        vec![1.0; dim],
        "Exact",
        true,
    );

    let clamped_cert =
        crate::pipeline::verify_pipeline(&[stage1, stage2, stage_clip]).expect("clamped pipeline");
    assert!(clamped_cert.is_valid);

    // P1-P3, P6 from clamped pipeline.
    let bundle = verify_properties_from_pipeline(&clamped_cert, dim);
    assert!(
        bundle.results[1].proven,
        "P2 (non-clipping) must pass with audio clamp"
    );

    let crates_dir = workspace_crates_dir();

    // P8: Dynamic ay kernel coverage + dispatch plan.
    let ay_names = ay_proven_kernel_names();
    let smt_evidence = SmtVerificationEvidence {
        kernels_proven: ay_names.len(),
        kernels_total: ay_names.len(),
        proven_kernel_names: ay_names.iter().map(ToString::to_string).collect(),
        all_proven: true,
    };
    let (kokoro_steps, _) = crate::kokoro_dispatch::build_kokoro_dispatch_plan_default();
    let dispatch_evidence = analyze_dispatch_plan(&kokoro_steps);

    let source_hash = crate::moonshot::compute_workspace_source_hash(
        crates_dir.parent().expect("workspace root"),
    )
    .unwrap_or_else(|_| "unavailable".to_string());

    // Build full 8-property certificate.
    let cert = FullCertificateBuilder::new(
        "kokoro-82m-d512-precomputed",
        "English text, D=512, IBP + audio clamp",
        &source_hash,
    )
    .crown_bundle(&bundle)
    .smt(&smt_evidence)
    .dispatch_plan(&dispatch_evidence)
    .build();

    assert_eq!(cert.properties.len(), 8);
    // IBP stages have is_sound=false, so pipeline cert is not sound.
    // check_non_clipping returns CrownPartial (proven=true, not CrownProven).
    assert!(
        matches!(
            cert.properties[1].level,
            VerificationLevel::CrownPartial
                | VerificationLevel::CrownProven
                | VerificationLevel::SmtProven
        ),
        "P2 must be at least CrownPartial with audio clamp, got {:?}",
        cert.properties[1].level
    );
    assert!(
        matches!(
            cert.properties[5].level,
            VerificationLevel::CrownPartial
                | VerificationLevel::CrownProven
                | VerificationLevel::SmtProven
        ),
        "P6 must be at least CrownPartial with audio clamp, got {:?}",
        cert.properties[5].level
    );
    assert_eq!(cert.properties[6].level, VerificationLevel::None, "P7");
    assert_eq!(cert.properties[7].level, VerificationLevel::SmtProven, "P8");

    eprintln!("\n=== D=512 Precomputed + Audio Clamp: 8-Property Certificate ===");
    for p in &cert.properties {
        let proven = matches!(
            p.level,
            VerificationLevel::CrownProven | VerificationLevel::SmtProven
        );
        eprintln!(
            "  P{}: {} — level={:?}, proven={}",
            p.property_index + 1,
            p.property_name,
            p.level,
            proven,
        );
    }
    eprintln!("All proven: {}", cert.all_proven);
}
